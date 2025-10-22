# Document Loader - Retrievel Document Generator
# Author: Muhammad Humayun Khan

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load the key and create model
load_dotenv()
model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Summarize the topic \n {poem}',
    input_variables = ['poem']
)

parser = StrOutputParser()

# upload the file
Loader = TextLoader("struggling_poem.txt",encoding='utf-8')
docs = Loader.load()    # save to the memory

chain = prompt | model | parser
print(chain.invoke({'poem':docs[0].page_content}))
print(docs)


