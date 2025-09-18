import os
import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

from .graph_loader import load_documents_and_create_graph, GraphLoadResult

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("graph-rag-app")

app = FastAPI(
    title="Graph RAG App",
    description="A React application interacting with a Graph-RAG backend.",
    version="0.3.0",
)

allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in allow_origins_env.split(",")] if allow_origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
app.state.llm = None
app.state.embedding_model = None
app.state.graph = None
app.state.vector_index_name = os.getenv("VECTOR_INDEX_NAME", "entity_embedding_index")
app.state.retriever = None
app.state.rag_chain = None
app.state.last_load_result: Optional[GraphLoadResult] = None
app.state.startup_timings = {}

REQUIRED_ENVS = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "HUGGING_FACE_HUB_TOKEN"]

def ensure_env() -> None:
    missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env(s): {', '.join(missing)}")

def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs):
        txt = d.page_content if hasattr(d, "page_content") else str(d)
        if len(txt) > 800:
            txt = txt[:800] + " ..."
        parts.append(f"[{i+1}] {txt}")
    return "\n\n".join(parts)

def build_rag_chain(vector_index_name: str):
    vecstore = Neo4jVector.from_existing_index(
        embedding=app.state.embedding_model,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        index_name=vector_index_name,
    )
    retriever = vecstore.as_retriever(search_kwargs={"k": int(os.getenv("TOPK", "6"))})
    app.state.retriever = retriever

    template = """Answer the question based ONLY on the following context.
If you are not sure, say you are not sure.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | app.state.llm
        | StrOutputParser()
    )
    app.state.rag_chain = rag_chain
    logger.info("RAG chain is ready.")

def create_vector_index_if_supported(index_name: str, dimensions: int = 384) -> None:
    index_query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:__Entity__) ON (n.embedding)
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """
    try:
        app.state.graph.query(index_query)
        logger.info("Vector index ensured/created.")
    except Exception as e:
        logger.warning(f"Vector index creation failed (maybe unsupported edition). Detail: {e}")

@app.on_event("startup")
def on_startup():
    t0_all = time.perf_counter()
    ensure_env()
    logger.info("Starting up...")

    # LLM
    t0 = time.perf_counter()
    logger.info("Initializing LLM pipeline manually to bypass library bug...")
    llm_pipeline = pipeline(
        task=os.getenv("HF_TASK", "text-generation"),
        model=os.getenv("HF_REPO", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device=os.getenv("LLM_DEVICE", "cpu"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "768")),
        do_sample=os.getenv("DO_SAMPLE", "False").lower() == "true",
        top_p=float(os.getenv("TOP_P", "1.0")),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
    )
    app.state.llm = HuggingFacePipeline(pipeline=llm_pipeline)
    app.state.startup_timings["llm_init_s"] = time.perf_counter() - t0
    logger.info(f"LLM initialized in {app.state.startup_timings['llm_init_s']:.2f}s")

    # Embeddings
    t0 = time.perf_counter()
    app.state.embedding_model = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
    )
    app.state.startup_timings["embedding_init_s"] = time.perf_counter() - t0
    logger.info(f"Embedding initialized in {app.state.startup_timings['embedding_init_s']:.2f}s")

    # Graph
    t0 = time.perf_counter()
    app.state.graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    app.state.startup_timings["neo4j_connect_s"] = time.perf_counter() - t0
    logger.info(f"Connected to Neo4j in {app.state.startup_timings['neo4j_connect_s']:.2f}s")

    embedding_dim = 384 # Default for all-MiniLM-L6-v2
    
    if os.getenv("INGEST_ON_STARTUP", "false").lower() == "true":
        logger.info("INGEST_ON_STARTUP is true, running data ingestion...")
        # Optional reset
        if os.getenv("EAGER_RESET_GRAPH", "false").lower() == "true":
            logger.warning("EAGER_RESET_GRAPH=true -> Clearing ALL nodes & relationships!")
            app.state.graph.query("MATCH (n) DETACH DELETE n")

        # Load docs → build graph
        t0 = time.perf_counter()
        load_dir = os.getenv("DOCS_DIR")  # if None -> ./docs
        result = load_documents_and_create_graph(app.state.llm, app.state.graph, folder_path=load_dir)
        app.state.last_load_result = result
        app.state.startup_timings["graph_build_s"] = time.perf_counter() - t0
        logger.info(f"Graph built in {app.state.startup_timings['graph_build_s']:.2f}s | stats={result.dict()}")
        if result:
            embedding_dim = result.embedding_dim
    else:
        logger.info("INGEST_ON_STARTUP is false, skipping data ingestion.")


    # Vector index
    t0 = time.perf_counter()
    create_vector_index_if_supported(app.state.vector_index_name, dimensions=embedding_dim)
    app.state.startup_timings["vector_index_s"] = time.perf_counter() - t0
    logger.info(f"Vector index ensured in {app.state.startup_timings['vector_index_s']:.2f}s")

    # RAG chain
    t0 = time.perf_counter()
    build_rag_chain(app.state.vector_index_name)
    app.state.startup_timings["rag_chain_init_s"] = time.perf_counter() - t0
    logger.info(f"RAG chain initialized in {app.state.startup_timings['rag_chain_init_s']:.2f}s")

    app.state.startup_timings["total_startup_s"] = time.perf_counter() - t0_all
    logger.info(f"Startup finished in {app.state.startup_timings['total_startup_s']:.2f}s")

class Question(BaseModel):
    question: str = Field(..., min_length=1)
    @validator("question")
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank")
        return v.strip()

@app.get("/health")
def health():
    try:
        cnt_entities = app.state.graph.query("MATCH (n:__Entity__) RETURN count(n) AS c")[0]["c"]
    except Exception:
        cnt_entities = None
    return {
        "status": "ok",
        "entities": cnt_entities,
        "last_load": app.state.last_load_result.dict() if app.state.last_load_result else None,
        "vector_index": app.state.vector_index_name,
        "allow_origins": allow_origins,
        "startup_timings": app.state.startup_timings,
    }

@app.post("/ask")
async def ask_question(item: Question):
    try:
        answer = await run_in_threadpool(app.state.rag_chain.invoke, item.question)
        return {"answer": answer}
    except Exception as e:
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_docs(x_admin_token: str = Header(default=None)):
    admin_token = os.getenv("ADMIN_TOKEN")
    if not admin_token or x_admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        t0 = time.perf_counter()
        result = await run_in_threadpool(
            load_documents_and_create_graph,
            app.state.llm,
            app.state.graph,
            os.getenv("DOCS_DIR"),
        )
        create_vector_index_if_supported(app.state.vector_index_name, dimensions=result.embedding_dim)
        build_rag_chain(app.state.vector_index_name)
        elapsed = time.perf_counter() - t0
        app.state.last_load_result = result
        return {"reloaded": True, "stats": result.dict(), "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        logger.exception("Error in /reload")
        raise HTTPException(status_code=500, detail=str(e))
