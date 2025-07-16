import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
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
            docs = loader.load()
            documents.extend(docs)
        elif file.endswith(".pdf"):
            # ใช้ pdfminer.six แปลง PDF เป็นข้อความ
            try:
                text = extract_text(file_path)
                if text:
                    # สร้างเอกสารในรูปแบบที่เหมาะสมกับ LangChain
                    from langchain.schema import Document
                    doc = Document(page_content=text, metadata={"source": file_path})
                    documents.append(doc)
                else:
                    print(f"❌ No text extracted from: {file}")
            except Exception as e:
                print(f"❌ Error extracting PDF {file}: {e}")
        elif file.endswith(".csv"):
            loader = CSVLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        else:
            print(f"❌ Unsupported file format: {file}")
            continue

    print(f"✅ Loaded {len(documents)} documents")

    # แบ่งเอกสาร
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    # สร้าง embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}
    )

    # สร้าง vector store (FAISS)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # (ถ้ามีการสร้าง triplets → ใส่ logic เพิ่มตรงนี้)
    print("✅ Vector store created (FAISS)")
    return vectorstore, graph

# ตัวอย่างการใช้ HuggingFaceEmbeddings แทน HuggingFaceBgeEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")