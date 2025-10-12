# Sequential Chain or Linear Chain
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load the key and the model
load_dotenv()
model = ChatOpenAI()

# the input taken by the LLM 
prompt_initial = PromptTemplate(
    template = 'Give me few sentences about {topic}',
    input_variables = ['topic']
)
prompt_two = PromptTemplate(
    template = 'Give me summary of the \n {text}',
    input_variables = {'text'}
)

parser = StrOutputParser()      # converting the input to plain text

chain = prompt_initial | model | parser | prompt_two | model | parser     # pipeline the different process automatically

result = chain.invoke({'topic':'AI Replace Jobs'})

print(result)

print(chain.get_graph().print_ascii())      # display in the graph form


