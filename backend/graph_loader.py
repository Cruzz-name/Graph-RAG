import os
import time
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def create_knowledge_graph_from_file(file_path):
    """
    Reads a text file, splits it into chunks, embeds each chunk,
    creates a knowledge graph using Llama3 via HuggingFacePipeline,
    and prints the resulting graph documents.
    """

    # 1️⃣ อ่านไฟล์
    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    # 2️⃣ แบ่งข้อความเป็น chunks (ลดขนาดต่อ chunk เพื่อลด error)
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    chunks = splitter.split_text(text_content)
    documents = [Document(page_content=chunk) for chunk in chunks]

    print(f"Total chunks to embed: {len(documents)}")

    # 3️⃣ สร้าง embedding model
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}  # ใช้ "cuda" ถ้ามี GPU
    )

    # 4️⃣ ทำ embedding แต่ละ chunk
    embedded_chunks = []
    print("\nEmbedding chunks...")
    for doc in tqdm(documents, desc="Embedding chunks"):
        try:
            vector = embeddings_model.embed_query(doc.page_content)
            embedded_chunks.append(vector)
            time.sleep(0.05)  # ลดความเร็วเล็กน้อย
        except Exception as e:
            print(f"Error embedding chunk: {e}")

    print(f"Successfully embedded {len(embedded_chunks)} chunks.")

    # 5️⃣ โหลด Llama 3 model ผ่าน HuggingFacePipeline
    print("\nLoading Llama 3 model...")
    model_name = "meta-llama/Llama-3-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",  # auto เลือก GPU ถ้ามี
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Llama 3 model loaded.")

    # 6️⃣ สร้าง LLMGraphTransformer
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 7️⃣ แปลงเอกสารแต่ละ chunk เป็น Graph Document ทีละ chunk
    print("\nCreating knowledge graph...")
    graph_documents = []
    for doc in tqdm(documents, desc="Creating graph from documents"):
        try:
            # ส่งทีละ Document เพื่อลด error
            graph_doc = llm_transformer.convert_to_graph_documents([doc])
            graph_documents.extend(graph_doc)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error converting to graph: {e}")

    print("\nKnowledge Graph created successfully.")
    print("\nGraph Documents Summary:")
    for i, doc in enumerate(graph_documents, start=1):
        print(f"\nGraph Document #{i}")
        print("Nodes:", doc.nodes)
        print("Relationships:", doc.relationships)
        print("-" * 40)

if __name__ == "__main__":
    file_path = "C:\\test_graph_rag\\Graph-RAG\\backend\\docs\\matcha.txt"
    create_knowledge_graph_from_file(file_path)
