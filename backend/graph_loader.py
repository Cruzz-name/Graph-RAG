import os
import json
import logging
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

GRAPH_GENERATION_PROMPT = """
คุณเป็นอัลกอริทึมสำหรับสกัดข้อมูลเชิงโครงสร้างจากข้อความภาษาไทย
จงสกัด "โหนด (nodes)" และ "ความสัมพันธ์ (relationships)" แล้วส่งออกเป็น JSON เดียวที่ถูกต้องเท่านั้น
(ห้ามมีคำอธิบายหรือโค้ดเฟนซ์ก่อน/หลัง JSON)

กติกา:
1) nodes: แต่ละโหนดต้องมี
   - "id": ชื่อเอนทิตี (คงรูปภาษาไทยตามต้นฉบับ)
   - "type": ประเภทเอนทิตี เช่น "บุคคล", "องค์กร", "สถานที่", "ผลิตภัณฑ์", "วัฒนธรรม", "สารอาหาร", "คุณประโยชน์", "คุณสมบัติ", "พิธีกรรม", "เมนู/สูตรอาหาร", "เครื่องสำอาง/สกินแคร์", "กระบวนการผลิต"
2) relationships: แต่ละรายการต้องมี
   - "source": id ของโหนดต้นทาง
   - "target": id ของโหนดปลายทาง
   - "type": ประเภทความสัมพันธ์ เช่น "กำเนิดที่", "เกี่ยวข้องกับ", "เป็นส่วนหนึ่งของ", "ประกอบด้วย", "ใช้ใน", "มีคุณประโยชน์ต่อ", "นิยมใน", "ใช้ผลิต", "เป็นสัญลักษณ์ของ"

ตัวอย่าง JSON:
{
  "nodes": [
    {"id": "มัทฉะ", "type": "ผลิตภัณฑ์"},
    {"id": "เกียวโต", "type": "สถานที่"}
  ],
  "relationships": [
    {"source": "มัทฉะ", "target": "เกียวโต", "type": "เกี่ยวข้องกับ"}
  ]
}

ข้อความ:
---
"""

@dataclass
class GraphLoadResult:
    doc_count: int
    chunk_count: int
    node_count: int
    rel_count: int
    embedding_dim: int = 384

    def dict(self):
        return asdict(self)

import re

from json_repair import loads as json_repair_loads

def extract_balanced_json(text: str) -> Optional[dict]:
    """
    Extracts a JSON object from a string by finding the first '{' and last '}'.
    This is more robust to variations in the LLM's output format.
    """
    try:
        # ค้นหาตำแหน่งเริ่มต้นของ JSON object
        json_start_index = text.find('{')
        if json_start_index == -1:
            logger.warning("Could not find start of JSON object ('{') in LLM response.")
            return None

        # ค้นหาตำแหน่งสุดท้ายของ JSON object
        json_end_index = text.rfind('}')
        if json_end_index == -1:
            logger.warning("Could not find end of JSON object ('}') in LLM response.")
            return None
        
        # ดึงข้อความส่วนที่เป็น JSON ออกมา
        json_str = text[json_start_index:json_end_index + 1]

        # ใช้ json_repair เพื่อซ่อมและแปลงข้อความ JSON ที่อาจไม่สมบูรณ์
        data = json_repair_loads(json_str)
        return data
    except Exception as e:
        logger.error(f"Failed to repair and decode JSON: {e}")
        logger.debug(f"Problematic text for JSON extraction: {text}")
        return None


def load_documents_and_create_graph(llm, graph: Neo4jGraph, folder_path: Optional[str] = None) -> GraphLoadResult:
    if not folder_path:
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

    logger.info("="*10 + " เริ่มฟังก์ชันการโหลดเอกสาร " + "="*10)
    logger.info(f"--- DEBUG: กำลังโหลดเอกสารจาก: {folder_path} ---")

    documents = []
    try:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(".txt"):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(Document(page_content=content, metadata={"source": file_path}))
                    except Exception as e:
                        logger.error(f"--- DEBUG: ไม่สามารถอ่านไฟล์ {file_path}: {e}")
    except Exception as e:
        logger.error(f"--- DEBUG: เกิดข้อผิดพลาดระหว่างการค้นหาไฟล์: {e}")


    logger.info(f"--- DEBUG: พบเอกสาร {len(documents)} ไฟล์ ---")

    # สร้าง instance ของตัวแบ่งข้อความ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    all_chunks = text_splitter.split_documents(documents)
    logger.info(f"--- DEBUG: แบ่ง {len(documents)} เอกสารออกเป็น {len(all_chunks)} ส่วนย่อย (chunks) ---")

    logger.info("="*50 + "\n")

    total_nodes = 0
    total_rels = 0

    for doc in tqdm(all_chunks, desc="กำลังประมวลผลส่วนย่อยของเอกสาร"):
        prompt = GRAPH_GENERATION_PROMPT + doc.page_content + "\n---\nJSON:\n"
        try:
            response_text = llm.invoke(prompt)
            logger.info(f"ผลลัพธ์จาก LLM สำหรับเอกสาร: {doc.metadata.get('source', 'Unknown')} ->\n---\n{response_text}\n---")
            graph_data = extract_balanced_json(response_text)

            if graph_data and "nodes" in graph_data and "relationships" in graph_data:
                # บันทึกข้อมูลกราฟที่สกัดได้ลงไฟล์ JSON เพื่อการตรวจสอบ
                json_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
                try:
                    with open(json_output_path, 'w', encoding='utf-8') as f:
                        json.dump(graph_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"บันทึกข้อมูลกราฟที่สกัดได้ลงใน {json_output_path} สำเร็จ")
                except Exception as e:
                    logger.error(f"ไม่สามารถบันทึกข้อมูลกราฟลงใน {json_output_path} ได้: {e}")

                # ตรวจสอบความถูกต้องและสร้างโหนด
                valid_nodes = []
                node_ids = set()
                for n in graph_data["nodes"]:
                    if n.get("id") and n.get("type"):
                        valid_nodes.append(Node(id=n["id"], label=n["type"]))
                        node_ids.add(n["id"])
                    else:
                        logger.warning(f"ข้ามโหนดที่ไม่ถูกต้อง: {n} ในเอกสาร: {doc.metadata.get('source', 'Unknown')}")

                # ตรวจสอบความถูกต้องและสร้างความสัมพันธ์
                valid_relationships = []
                for r in graph_data["relationships"]:
                    source_id = r.get("source")
                    target_id = r.get("target")
                    if source_id and target_id and r.get("type"):
                        if source_id in node_ids and target_id in node_ids:
                            valid_relationships.append(Relationship(
                                source=Node(id=source_id),
                                target=Node(id=target_id),
                                type=r["type"]
                            ))
                        else:
                            logger.warning(f"ข้ามความสัมพันธ์ที่โหนดหายไป: {r} ในเอกสาร: {doc.metadata.get('source', 'Unknown')}")
                    else:
                        logger.warning(f"ข้ามความสัมพันธ์ที่ไม่ถูกต้อง: {r} ในเอกสาร: {doc.metadata.get('source', 'Unknown')}")

                if valid_nodes:
                    graph_doc = GraphDocument(
                        nodes=valid_nodes, 
                        relationships=valid_relationships, 
                        source=doc
                    )
                    graph.add_graph_documents(
                        [graph_doc],
                        include_source=False
                    )
                    total_nodes += len(valid_nodes)
                    total_rels += len(valid_relationships)
                else:
                    logger.warning(f"ไม่พบโหนดที่ถูกต้องในเอกสาร: {doc.metadata.get('source', 'Unknown')}")
            else:
                logger.warning(f"ไม่สามารถสกัดกราฟที่ถูกต้องจากเอกสาร: {doc.metadata.get('source', 'Unknown')}")

        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการประมวลผลเอกสาร {doc.metadata.get('source', 'Unknown')}: {e}", exc_info=True)

    return GraphLoadResult(
        doc_count=len(documents),
        chunk_count=len(all_chunks),
        node_count=total_nodes,
        rel_count=total_rels,
    )