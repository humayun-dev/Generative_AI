# Runnable Sequence means to connect different processes into one
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate

# load the key and model
load_dotenv()

model = ChatOpenAI()

# create a prompt
prompt = PromptTemplate(
    template = 'One sentence on the {topic}',
    input_variables = ['topic']
)

# parse the output
parser = StrOutputParser()

# Call runnable sequence
chain = RunnableSequence(prompt,model,parser)

print(chain.invoke({'topic':'Struggling Life'}))