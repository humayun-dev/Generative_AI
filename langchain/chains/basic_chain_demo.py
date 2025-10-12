# Chain is used to automate the process of automation by avoiding manual process such as we will provide the input/prompt and then the output of the prompt/input will proceed to the LLM/model and the output of the model will be converted to the string by using the parser so all the process will be in the chaining and not acting manually
# we have three types of chains such as 
# 1) Sequential or linear
# 2) Parallel chains
# 3) Conditional chain

# first will implement the simple chain concept
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load the key and the model
load_dotenv()
model = ChatOpenAI()

# the input taken by the LLM 
prompt = PromptTemplate(
    template = 'Give me few sentences about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()      # converting the input to plain text

chain = prompt | model | parser     # pipeline the different process automatically

result = chain.invoke({'topic':'AI Replace Jobs'})

print(result)

print(chain.get_graph().print_ascii())      # display in the graph form


