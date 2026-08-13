from pathlib import Path
import json,pickle,re,os
import numpy as np
import faiss
from joblib import load
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from product_image_query.predict_product import predict as classify_product_image
ROOT=Path(__file__).resolve().parents[1]
MODEL=load(ROOT/'models/return_risk_model.pkl')
T_RF=json.loads((ROOT/'models/return_risk_threshold.json').read_text())['t_rf']
INDEX=faiss.read_index(str(ROOT/'part3/policy.index'))
CHUNKS=pickle.loads((ROOT/'part3/chunks.pkl').read_bytes())
EMBED=SentenceTransformer('all-MiniLM-L6-v2')

class State(TypedDict, total=False):
    messages: List[Dict[str,str]]
    order_id: Optional[str]
    intent: str
    retrieved: List[Dict[str,Any]]
    tool_output: Dict[str,Any]
    answer: Dict[str,Any]
    blocked: bool

INJECTION=re.compile(r'ignore\s+(previous|all)\s+(instructions|rules)|pretend\s+you\s+are|disregard\s+(previous|all)',re.I)

def intent_node(s):
    text=s['messages'][-1]['content'].lower()
    if INJECTION.search(text): return {'blocked':True,'intent':'blocked'}
    if any(k in text for k in ['return risk','risk of return','likely to return','return probability']): intent='return_risk'
    elif any(k in text for k in ['image','product category','what category','classify']): intent='product_category'
    else: intent='policy'
    m=re.search(r'order\s*#?\s*([A-Za-z0-9-]+)',s['messages'][-1]['content'],re.I)
    return {'intent':intent,'order_id':m.group(1) if m else s.get('order_id')}

def retrieve_node(s):
    if s.get('intent')!='policy': return {'retrieved':[]}
    q=s['messages'][-1]['content']; v=EMBED.encode([q],normalize_embeddings=True).astype('float32'); scores,ids=INDEX.search(v,3)
    return {'retrieved':[{'score':float(scores[0][i]),**CHUNKS[int(ids[0][i])]} for i in range(3)]}

def check_return_risk(order_features):
    import pandas as pd
    p=float(MODEL.predict_proba(pd.DataFrame([order_features]))[0,1]);
    bucket='Low' if p<T_RF else ('High' if p>=T_RF+0.15 else 'Medium')
    return {'probability':p,'risk_bucket':bucket,'t_rf':T_RF,'cut_points':{'low_lt':T_RF,'high_gte':T_RF+0.15}}

def tool_node(s):
    if s.get('intent')=='return_risk':
        # Features may be supplied in the last message as JSON after FEATURES:.
        text=s['messages'][-1]['content']; m=re.search(r'FEATURES:\s*(\{.*\})',text,re.S)
        if not m: return {'tool_output':{'error':'Provide order features as JSON after FEATURES:'}}
        return {'tool_output':check_return_risk(json.loads(m.group(1)))}
    if s.get('intent')=='product_category':
        m=re.search(r'IMAGE:\s*(\S+)',s['messages'][-1]['content'])
        return {'tool_output':classify_product_image(str(ROOT/m.group(1)) if m and not Path(m.group(1)).is_absolute() else m.group(1)) if m else {'error':'Provide IMAGE: path'}}
    return {'tool_output':{}}

def response_node(s):
    if s.get('blocked'): ans={'answer':'I can’t follow instructions that attempt to override the support assistant’s rules. Please ask a normal support question.','source':'policy_kb','confidence':1.0}
    elif s.get('intent')=='policy':
        good=[x for x in s.get('retrieved',[]) if x['score']>=0.35]
        if not good: ans={'answer':'I don’t have enough grounded information in the policy knowledge base to answer that reliably.','source':'policy_kb','confidence':0.0}
        else: ans={'answer':' '.join(x['text'] for x in good[:2]),'source':'policy_kb','confidence':float(good[0]['score'])}
    elif s.get('intent')=='return_risk':
        o=s.get('tool_output',{}); ans={'answer':f"The predicted return probability is {o.get('probability',0):.3f}, which is a {o.get('risk_bucket','Unknown')} risk. The bucket is anchored to t*_rf={T_RF:.2f}.",'source':'return_risk_tool','confidence':float(o.get('probability',0))}
    else:
        o=s.get('tool_output',{}); ans={'answer':f"The predicted product category is {o.get('category','Unknown')} with confidence {o.get('confidence',0):.3f}.",'source':'image_classifier_tool','confidence':float(o.get('confidence',0))}
    return {'answer':ans}

def route(s): return s.get('intent','policy')

g=StateGraph(State); g.add_node('intent',intent_node); g.add_node('retrieve',retrieve_node); g.add_node('tools',tool_node); g.add_node('response',response_node)
g.add_edge(START,'intent'); g.add_conditional_edges('intent',route,{'policy':'retrieve','return_risk':'tools','product_category':'tools','blocked':'response'}); g.add_edge('retrieve','response'); g.add_edge('tools','response'); g.add_edge('response',END)
GRAPH=g.compile()

def run(messages,order_id=None): return GRAPH.invoke({'messages':messages,'order_id':order_id})
