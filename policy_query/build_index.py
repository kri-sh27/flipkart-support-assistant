from pathlib import Path
import json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
KB=json.loads((ROOT/'part3/policy_kb.json').read_text())
chunks=[]
for doc in KB:
    # Sentence-wise chunking, retaining parent document id.
    for i,s in enumerate([x.strip() for x in doc['text'].replace('!','.!').replace('?','.?').split('.') if x.strip()]):
        chunks.append({'chunk_id':f"{doc['id']}_{i}",'doc_id':doc['id'],'text':s+'.'})
model=SentenceTransformer('all-MiniLM-L6-v2')
emb=model.encode([c['text'] for c in chunks],normalize_embeddings=True).astype('float32')
index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
faiss.write_index(index,str(ROOT/'part3/policy.index'))
(ROOT/'part3/chunks.pkl').write_bytes(pickle.dumps(chunks))
print('chunks:',len(chunks),'dimension:',emb.shape[1])
