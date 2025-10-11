# As the LLM returns two types of output such as structured and parser
# The structured output is beneficial for data extraction, API building and for the Agents
# The text generated is not the structured but the structred output is that of JSON format having key and value
# For using the structured, there is a class with_structured_output and that class having three types such as
# typedict = instructions for the data either integer, string etc. but user can voilate the instructions and code will run
# pydantic = this is used for the data validation
# JSON_schema

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal

# load the key
load_dotenv()

# load the model
model = ChatOpenAI()

# create schema using typedict

# class ProductReview(TypedDict):
#     summary: str
#     sentiment: str

# The annotated can also be used in which we provide the instructions to the LLM as what we want 
class ProductReview(TypedDict):
    summary: Annotated[str,"Brief summary of the review"]
    sentiment: Annotated[Literal['pos','neg','neutral'],"Review should be either positive, negative or neutral"]

structured_model = model.with_structured_output(ProductReview)

result = structured_model.invoke("The hardware of the Infinix is good but the software of the pixel is not good and its battery is also short in comparison of the infinix")

print(result)