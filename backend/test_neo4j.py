from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("✅ Connected to Neo4j!")
print("Sample Nodes:")
print(graph.query("MATCH (n) RETURN n LIMIT 5"))
