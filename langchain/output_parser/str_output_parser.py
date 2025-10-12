# Output Parser is used to convert raw response of LLM into structured formats like JSON, CSV, pydantic models etc
# There are four common used output parsers such as
# i. String output parser (take LLM response and convert it into plain text)
# ii. JSON output parser
# iii. Structured output parser
# iv. Pydantic output parser

# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load the key
load_dotenv()

# load the model
model = ChatOpenAI()

template_one = PromptTemplate(
    template = 'Write a summary on the {topic}',
    input_variables=['topic']
)
template_two = PromptTemplate(
    template = 'Write detail {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# chain pipeline which is template to the model, then parse it, then template 2 to the model and then parse it
chain = template_one | model | parser | template_two | model | parser

result = chain.invoke({'topic':'AI will replace jobs or not?'})
print(result)

