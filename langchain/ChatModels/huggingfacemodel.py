# Using open source hugging face model text generation of NLP
# Author: Muhammad Humayun Khan

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# select model and its task
llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("What is the capital of Pakistan?")

print(result.content)

