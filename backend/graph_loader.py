import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    CSVLoader
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# เชื่อมต่อ Neo4j
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

# ตัวอย่าง path เอกสารในเครื่อง
docs_path = "./docs"

# ฟังก์ชันโหลดเอกสารและสร้าง Knowledge Graph
def load_documents_and_create_graph(llm, graph, folder_path="docs"):
    documents = []
    # ใช้ folder_path เพื่อโหลดเอกสาร
    loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    documents.extend(docs)

    for file in os.listdir(docs_path):
        file_path = os.path.join(docs_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path)
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            print(f"❌ Unsupported file format: {file}")
            continue

        docs = loader.load()
        documents.extend(docs)

    print(f"✅ Loaded {len(documents)} documents")

    # แบ่งเอกสาร
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    # สร้าง embedding model
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )

    # สร้าง vector store (FAISS)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # (ถ้ามีการสร้าง triplets → ใส่ logic เพิ่มตรงนี้)
    print("✅ Vector store created (FAISS)")
    return vectorstore, graph
