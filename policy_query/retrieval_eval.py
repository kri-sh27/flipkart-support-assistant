from pathlib import Path
import json,pickle
import numpy as np,faiss
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
queries=[
('How long can I return footwear?',['policy_apparel_footwear_returns']),
('When should I expect a COD refund?',['policy_cod_refunds']),
('What is the normal delivery time?',['policy_delivery_sla']),
('Can you collect my return from my address?',['policy_reverse_pickup']),
('What happens if an electronics item has a defect?',['policy_electronics_returns'])]
chunks=pickle.loads((ROOT/'part3/chunks.pkl').read_bytes()); idx=faiss.read_index(str(ROOT/'part3/policy.index')); model=SentenceTransformer('all-MiniLM-L6-v2')
out=[]
for q,gold in queries:
 v=model.encode([q],normalize_embeddings=True).astype('float32'); scores,ids=idx.search(v,3); docs=[]
 for i in ids[0]:
  d=chunks[int(i)]['doc_id']
  if d not in docs: docs.append(d)
 hit=set(docs); g=set(gold); p=len(hit&g)/3; r=len(hit&g)/len(g)
 out.append({'query':q,'gold':gold,'retrieved_documents':docs,'precision_at_3':p,'recall_at_3':r})
result={'per_query':out,'average_precision_at_3':float(np.mean([x['precision_at_3'] for x in out])),'average_recall_at_3':float(np.mean([x['recall_at_3'] for x in out]))}
(ROOT/'results/retrieval_eval.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
