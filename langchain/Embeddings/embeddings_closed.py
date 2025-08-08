# Embeddings using OpenAI API
# the program finds out embeddings for a single string and also for the document
# Author: Muhammad Humayun Khan

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large",dimensions = 32)

# embeddings of a single string
result = embeddings.embed_query("Capital of Pakistan is Islamabad")

print(str(result))

print("\n\n")
print("The document embeddings are as follow:")

# Embeddings for a document
document = [
    "This is Pakistan",
    "This is becoming IT sector",
    "This is growing country"
]

result = embeddings.embed_documents(document)
print(str(result))
