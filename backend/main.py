# main.py 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph_loader import load_documents_and_create_graph
from langchain.chat_models import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
import os

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# LLM & Graph Setup
llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Build graph from data/
load_documents_and_create_graph(llm, graph, folder_path="docs")
qa_chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)

# API model
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(item: Question):
    result = qa_chain.run(item.question)
    return {"answer": result}
