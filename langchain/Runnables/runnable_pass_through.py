# Runnable Pass Through - it receives the input and forward it as it is as output
# Sometimes we need to input any topic and generate something from the input topic. Now we have the generated output
# in the chain and we don't have the original topic in the chain as it has been modified, so in order to get the
# original topic along with the generated one, we use the runnable pass through.
# Author: Muhammad Humayun khan


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# load the key and model
load_dotenv()

model = ChatOpenAI()

# create two prompts
prompt = PromptTemplate(
    template = 'One sentence on the {topic}',
    input_variables = ['topic']
)

prompt_two = PromptTemplate(
    template = 'Explain the topic {topic}',
    input_variables = {'topic'}
)

# parse the output
parser = StrOutputParser()

# First part of the chain where we have to generate the topic
topic_generated = RunnableSequence(prompt,model,parser)

# call runnableParallel which will return output as dict and keys are topic_name and topic_explain
parallel_chain = RunnableParallel({
    'topic_name':RunnablePassthrough(),
    'topic_explain':RunnableSequence(prompt_two,model,parser)
})

final_chain = RunnableSequence(topic_generated,parallel_chain)

result = final_chain.invoke({'topic':'Struggling Life'})

print(result)