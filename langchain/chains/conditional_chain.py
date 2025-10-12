# Conditional chain in which the model behaves on any kind of condition
# for example, analyze the feedback of the consumer by sending it to the model
# The model will analyze the feedback and will result as either positive or negative sentiment
# The positive sentiment will be send to the model and the model will react and vicer versa
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# load the keys and model
load_dotenv()
model = ChatOpenAI()

parser = StrOutputParser()

# this class is data validation as we need positive/negative words only not anything else
class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser_two = PydanticOutputParser(pydantic_object=Feedback)

# classification prompt and its output should only be positive or negative
# in order to make its output only the desired one we need to implement data validation
prompt = PromptTemplate(
    template = 'Classify the sentiment of the following feedback into positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser_two.get_format_instructions()}
)

# classifier chain
classifier_chain = prompt | model | parser_two

# checking either sentiment working or not
# result = classifier_chain.invoke({'feedback':'This is a wonderful smart phone'}).sentiment
# print(result)

# now as there are conditions such as positive/negative so we will use the branch chain
# syntax for branch chain is condition_one,chain, condition_two,chain etc
prompt_condition_one = PromptTemplate(
    template='Write an appropriate response to the positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt_condition_two = PromptTemplate(
    template = 'Write an appropriate response to the negative feedback \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt_condition_one | model | parser),
    (lambda x:x.sentiment == 'negative', prompt_condition_two | model | parser),
    RunnableLambda(lambda x: "Could not find any kind of sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'this is a wonderful phone'}))


