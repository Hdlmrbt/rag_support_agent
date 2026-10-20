from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()  # charge OPENAI_API_KEY depuis .env

# 1. Configuration globale
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=50)

# 2. Charger les documents
documents = SimpleDirectoryReader("data").load_data()
print(f"{len(documents)} documents chargés")

# 3. Connexion à Qdrant
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="support_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Découper, vectoriser, stocker
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)
print("Ingestion terminée")