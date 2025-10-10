# A try for the chatbot
# for storing the user history, three types of builtin messages will be used
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage    
from dotenv import load_dotenv

# load the key
load_dotenv()

# create the model object
model = ChatOpenAI()

# store user previous conversation
user_chat = [
    SystemMessage(content='Act like a AI Assistant')
]

while True:
    user_input = input("User: ")

    if user_input == 'exit':
        break
    
    user_chat.append(HumanMessage(content=user_input))
    
    ai_response = model.invoke(user_chat)
    user_chat.append(ai_response)
    print("Chatbot: ",ai_response.content)

print(user_chat)