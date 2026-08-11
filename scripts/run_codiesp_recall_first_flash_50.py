#!/usr/bin/env python
from __future__ import annotations

import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import run_codiesp_direct_deepseek_thinking as base

MODEL='deepseek-v4-flash'
PROMPT='''You are an expert clinical coding specialist performing the CodiEsp-D diagnosis-coding task. Your primary goal is HIGH RECALL WITHOUT INVENTING DIAGNOSES: recover the complete set of ICD-10-CM/CIE10 diagnosis codes supported anywhere in the full clinical case.

IMPORTANT TASK INTERPRETATION:
- Do NOT code only the main diagnosis or discharge diagnosis.
- Do NOT stop after a short list. A single CodiEsp clinical case can contain many codable diagnosis entities.
- Systematically cover diagnosis information across the ENTIRE document: presenting diseases, past/current comorbidities, etiologies, complications, manifestations, infections, neoplasms/metastases, pathology-confirmed entities, adverse effects, and explicitly documented symptoms/signs or abnormal conditions when they are independently codable diagnosis entities.
- Do not omit a diagnosis merely because it is secondary, appears late in the narrative, or is not the main reason for care.
- Exclude procedures/tests/treatments, pure differential or ruled-out/negated diagnoses, and unsupported inference.
- Use the most specific ICD-10 diagnosis code justified by the text. Codes may be 3-character categories or more specific subcodes; choose the most specific supported code.

INTERNAL COMPLETENESS PROCEDURE BEFORE FINAL ANSWER:
1. Scan the text sentence by sentence and build a private list of every explicit diagnostic entity/condition.
2. Map EACH retained entity to an ICD-10 diagnosis code.
3. Re-scan Past Medical History/background, pathology/imaging conclusions, complications, and final assessment for missed diagnoses.
4. Compare the diagnosis-entity list against the code list one-by-one. Do not finish while a supported diagnosis entity remains uncoded.
5. Then remove only codes that truly lack textual support.

Return strict JSON only:
{"codes":["CODE1","CODE2"],"items":[{"code":"CODE1","diagnosis":"normalized diagnosis","evidence":"short exact phrase copied from the case"}]}.

CLINICAL CASE:
{case}'''

def call(prompt,key):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Perform exhaustive, evidence-grounded ICD-10 diagnosis coding. Think carefully and optimize for complete supported code recovery.'},{'role':'user','content':prompt}], 'thinking':{'type':'enabled'},'reasoning_effort':'high','response_format':{'type':'json_object'},'max_tokens':7000}).encode()
    last=None
    for attempt in range(4):
        try:
            req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=body,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=300) as r: payload=json.loads(r.read().decode())
            text=payload['choices'][0]['message']['content'].strip(); text=re.sub(r'^```(?:json)?\s*|\s*```$','',text,flags=re.I); return json.loads(text)
        except Exception as exc: last=exc; time.sleep(2**attempt)
    raise RuntimeError(str(last))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--n',type=int,default=50); ap.add_argument('--workers',type=int,default=10); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); key=os.getenv('DEEPSEEK_API_KEY','').strip()
    if not key: raise RuntimeError('DEEPSEEK_API_KEY required')
    _,texts,_,gold=base.load_codiesp(); ids=sorted(set(texts)&set(gold))[:args.n]
    def one(i):
        obj=call(PROMPT.replace('{case}',texts[i]),key); codes={base.normalize_code(x) for x in obj.get('codes',[]) if base.normalize_code(x)}; items=obj.get('items',[]) if isinstance(obj.get('items',[]),list) else []; return i,codes,items
    pred={}; rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(one,i):i for i in ids}
        for k,f in enumerate(as_completed(futs),1):
            i=futs[f]
            try:
                i,codes,items=f.result(); pred[i]=codes; rows.append({'article_id':i,'gold':'|'.join(sorted(gold[i])),'pred':'|'.join(sorted(codes)),'items_json':json.dumps(items,ensure_ascii=False)})
            except Exception as exc: pred[i]=set(); failures.append({'article_id':i,'error':str(exc)})
            if k%10==0: print(f'{k}/{len(ids)}',flush=True)
    rows.sort(key=lambda x:x['article_id']); pd.DataFrame(rows).to_csv(out/'recall_first_predictions.csv',index=False); pd.DataFrame(failures).to_csv(out/'failures.csv',index=False)
    m=base.metrics(gold,pred,ids); m['n_failed_requests']=len(failures)
    summary={'model':MODEL,'thinking':{'type':'enabled','reasoning_effort':'high'},'prompt_variant':'codiesp_recall_first','no_retrieval_or_candidate_list':True,'subset':'first 50 sorted test IDs, same as prior comparisons','metrics':m,'prompt':PROMPT}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    if failures: raise RuntimeError(f'{len(failures)} failures')
if __name__=='__main__': main()
