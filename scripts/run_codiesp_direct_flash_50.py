#!/usr/bin/env python
from __future__ import annotations

import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import run_codiesp_direct_deepseek_thinking as base

MODEL='deepseek-v4-flash'

def call(prompt,key):
    body=json.dumps({'model':MODEL,'messages':[{'role':'system','content':'Return clinically grounded ICD-10 diagnosis coding as strict JSON. Think carefully before the final answer.'},{'role':'user','content':prompt}], 'thinking':{'type':'enabled'},'reasoning_effort':'high','response_format':{'type':'json_object'},'max_tokens':5000}).encode()
    last=None
    for attempt in range(4):
        try:
            req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=body,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=240) as r: payload=json.loads(r.read().decode())
            text=payload['choices'][0]['message']['content'].strip(); text=re.sub(r'^```(?:json)?\s*|\s*```$','',text,flags=re.I)
            return json.loads(text)
        except Exception as exc:
            last=exc; time.sleep(2**attempt)
    raise RuntimeError(str(last))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--n',type=int,default=50); ap.add_argument('--workers',type=int,default=10); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); key=os.getenv('DEEPSEEK_API_KEY','').strip()
    if not key: raise RuntimeError('DEEPSEEK_API_KEY required')
    _,texts,_,gold=base.load_codiesp(); ids=sorted(set(texts)&set(gold))[:args.n]; template=base.PROMPTS['direct_exhaustive']
    def one(i):
        obj=call(template.replace('{case}',texts[i]),key); codes={base.normalize_code(x) for x in obj.get('codes',[]) if base.normalize_code(x)}; items=obj.get('items',[]) if isinstance(obj.get('items',[]),list) else []; return i,codes,items
    pred={}; rows=[]; fail=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(one,i):i for i in ids}
        for k,f in enumerate(as_completed(futs),1):
            i=futs[f]
            try:
                i,codes,items=f.result(); pred[i]=codes; rows.append({'article_id':i,'gold':'|'.join(sorted(gold[i])),'pred':'|'.join(sorted(codes)),'items_json':json.dumps(items,ensure_ascii=False)})
            except Exception as e: pred[i]=set(); fail.append({'article_id':i,'error':str(e)})
            if k%10==0: print(f'{k}/{len(ids)}',flush=True)
    rows.sort(key=lambda x:x['article_id']); pd.DataFrame(rows).to_csv(out/'direct_flash_50_predictions.csv',index=False); pd.DataFrame(fail).to_csv(out/'failures.csv',index=False)
    m=base.metrics(gold,pred,ids); m['n_failed_requests']=len(fail)
    summary={'model':MODEL,'thinking':{'type':'enabled','reasoning_effort':'high'},'prompt_variant':'direct_exhaustive','no_retrieval_or_candidate_list':True,'subset':'first 50 sorted test IDs','metrics':m,'prompt':template}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    if fail: raise RuntimeError(f'{len(fail)} failures')
if __name__=='__main__': main()
