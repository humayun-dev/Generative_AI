# Embeddings using hugging face API - The model used will be downloaded locally and almost size of 241 MB
# Author: Muhammad Humayun Khan

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vectors = embeddings.embed_query(embeddings)
print(str(vectors))