#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import run_codiesp_direct_deepseek_thinking as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    key = os.getenv('DEEPSEEK_API_KEY','').strip()
    if not key: raise RuntimeError('DEEPSEEK_API_KEY required')
    _, test_text, _, test_gold = base.load_codiesp()
    ids = sorted(set(test_text) & set(test_gold))[:args.n]
    template = base.PROMPTS['direct_exhaustive']

    def one(doc_id):
        prompt = template.replace('{case}', test_text[doc_id])
        obj = base.call_deepseek(prompt, key)
        codes = {base.normalize_code(c) for c in obj.get('codes',[]) if base.normalize_code(c)}
        items = obj.get('items',[]) if isinstance(obj.get('items',[]),list) else []
        return doc_id, codes, items

    pred={}; rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(one,i):i for i in ids}
        for k,f in enumerate(as_completed(futs),start=1):
            i=futs[f]
            try:
                doc_id,codes,items=f.result(); pred[doc_id]=codes
                rows.append({'article_id':doc_id,'gold':'|'.join(sorted(test_gold[doc_id])),'pred':'|'.join(sorted(codes)),'items_json':json.dumps(items,ensure_ascii=False)})
            except Exception as exc:
                pred[i]=set(); failures.append({'article_id':i,'error':str(exc)})
            if k%10==0: print(f'completed {k}/{len(ids)}',flush=True)
    rows.sort(key=lambda r:r['article_id']); failures.sort(key=lambda r:r['article_id'])
    pd.DataFrame(rows).to_csv(out/'direct_thinking_50_predictions.csv',index=False)
    pd.DataFrame(failures).to_csv(out/'direct_thinking_50_failures.csv',index=False)
    m=base.metrics(test_gold,pred,ids); m['n_failed_requests']=len(failures)
    summary={
      'schema_version':'codiesp-direct-thinking-50-v0.1','model':base.MODEL,
      'thinking':{'type':'enabled','reasoning_effort':'max'},'prompt_variant':'direct_exhaustive',
      'no_retrieval_or_candidate_list':True,'subset':'first 50 sorted test IDs, identical to prior DeepSeek candidate-assisted subset',
      'metrics':m,'prompt':template
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    if failures: raise RuntimeError(f'{len(failures)} API failures')

if __name__=='__main__': main()
