# api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import os
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from common import rerank_documents

from langchain_community.chat_models import ChatOllama

from langchain_community.embeddings import HuggingFaceEmbeddings

import pickle

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to your RAG API!"}

# Initialize components once (for performance)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retriever = FAISS.load_local("faiss_index", embedding_function, allow_dangerous_deserialization=True).as_retriever(search_kwargs={'k': 10})


llm = ChatOllama(model="mistral")


prompt = ChatPromptTemplate.from_template("""
[INST] You are a concise assistant. Answer the question using only the provided context. 
If the answer is not in the context, respond with "I don't know".

Context:
{context}

Question:
{input}

Answer: [/INST]
""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Input schema
class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: AskRequest):
    query = request.query
    start_time = time.time()

    

    try:
        docs = retriever.invoke(query)

        print("RETRIEVED DOCS TYPE ===>", type(docs), type(docs[0]))

        top_docs = rerank_documents(query, docs, top_n=5)

        print("TOP DOCS DEBUG ===>")
        for doc in top_docs:
            print(type(doc), getattr(doc, "page_content", "NO page_content"))

        response = document_chain.invoke({"context": top_docs, "input": query})
        latency = round(time.time() - start_time, 2)

        return {
            "query": query,
            "answer": response,
            "top_chunks": [doc.page_content for doc in top_docs],
            "metadata": [doc.metadata for doc in top_docs],
            "latency": latency
        }



    except Exception as e:
        return {"error": str(e)}

