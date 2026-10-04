import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

load_dotenv()
app = Flask(__name__)
CORS(app)
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
GROQ_API_KEY=os.getenv('GROQ_API_KEY')
INDEX_NAME=os.getenv('PINECONE_INDEX_NAME','medical-chatbot')
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise RuntimeError('Set PINECONE_API_KEY and GROQ_API_KEY in backend/.env')
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
pc=Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Pinecone index '{INDEX_NAME}' was not found.")
docsearch=PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
llm=ChatGroq(model='openai/gpt-oss-20b',temperature=0.1)
SYSTEM='''You are a knowledgeable medical information assistant. Answer using the retrieved medical reference below. Give useful, detailed answers; do not limit answers to three sentences. Start with a direct explanation, then use clear headings, bullets, and tables when useful. Explain difficult medical terms simply. Do not invent facts. If the reference does not contain enough information, say so. Do not diagnose the user or replace professional medical advice. Include causes, symptoms, diagnosis, complications, treatment or prevention only when relevant. If asked for an exam answer, make it structured and easy to memorize.\n\nRetrieved medical context:\n{context}'''
prompt=ChatPromptTemplate.from_messages([('system',SYSTEM),('human','{input}')])

def retrieve(q):
    try: return docsearch.similarity_search_with_score(q,k=6)
    except Exception: return [(d,None) for d in docsearch.similarity_search(q,k=6)]

def answer(q,history=[]):
    results=retrieve(q)
    context='\n\n'.join(f"[Source {i} — Page {d.metadata.get('page')+1 if isinstance(d.metadata.get('page'),int) else 'unknown'}]\n{d.page_content}" for i,(d,s) in enumerate(results,1))
    recent='\nRecent conversation:\n'+'\n'.join(f"{m.get('role','user').title()}: {m.get('content','')}" for m in history[-6:] if m.get('content')) if history else ''
    r=llm.invoke(prompt.invoke({'context':context,'input':q+recent}))
    sources=[]
    for i,(d,s) in enumerate(results,1):
        page=d.metadata.get('page')
        sources.append({'id':i,'page':page+1 if isinstance(page,int) else None,'score':round(float(s),4) if s is not None else None})
    return {'answer':r.content,'sources':sources}

@app.get('/api/health')
def health(): return jsonify({'status':'ok'})

@app.post('/api/chat')
def chat():
    data=request.get_json(silent=True) or {}; q=(data.get('message') or '').strip()
    if not q: return jsonify({'error':'Message is required.'}),400
    try: return jsonify(answer(q,data.get('history') or []))
    except Exception:
        app.logger.exception('RAG request failed'); return jsonify({'error':'Unable to process the question right now.'}),500

@app.post('/api/regenerate')
def regenerate():
    data=request.get_json(silent=True) or {}; q=(data.get('message') or '').strip()
    if not q: return jsonify({'error':'Message is required.'}),400
    try: return jsonify(answer(q,data.get('history') or []))
    except Exception:
        app.logger.exception('Regeneration failed'); return jsonify({'error':'Unable to regenerate the response right now.'}),500

if __name__=='__main__': app.run(host='0.0.0.0',port=5000,debug=True)
