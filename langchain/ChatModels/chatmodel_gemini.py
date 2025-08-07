# Chat Model using the Google Gemini API Key
# Author: Muhammad Humayun Khan

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Get the Gemini API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the key is not None
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Create Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",      
    google_api_key=api_key      
)

# Send a prompt to the model
response = model.invoke("What is the capital of Pakistan?")

# Print the response
#print(response.content)
print(response.response_metadata)  # Might show token usage
