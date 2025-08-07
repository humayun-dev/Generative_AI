# Basic demo of Chat Models using OpenAI API
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = "gpt-4")
result = model.invoke("What is the capital of Pakistan")

print(result) # print result along with all the arguments
print(result.content) # print result of the content only


# temperature value range from 0 to 2, 0->0.3 is for factual answers such as coding, maths, fact
# 0.5->0.7 is for balanced response such as general Q/A, explanation
# 0.9->1.2 is for creating writings, storytelling and jokes
# 1.5+ is for maximum randomness such as brainstorming and ideas
model = ChatOpenAI(model = "gpt-4",temperature=0,max_completion_tokens=15)
result = model.invoke("Write poem on the soccer")
print(result.content)