from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="support_docs")
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(similarity_top_k=4)

question = input("Question : ")
response = query_engine.query(question)

print("\n=== RÉPONSE ===")
print(response)

print("\n=== CHUNKS UTILISÉS ===")
for i, node in enumerate(response.source_nodes, 1):
    print(f"[{i}] {node.metadata.get('file_name')} (score: {node.score:.3f})")
    print(f"    {node.text[:150]}...\n")