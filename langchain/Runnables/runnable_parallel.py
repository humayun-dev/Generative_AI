# Runnable Parallel - to execute different runnables in parallel
# The input will be the same for both the LLMs
# The result is returned as Dictionary format
# Author: Muhammad Humayun khan


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel
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

# call runnableParallel which will return output as dict and keys are topic_name and topic_explain
parallel_chain = RunnableParallel({
    'topic_name':RunnableSequence(prompt,model,parser),
    'topic_explain':RunnableSequence(prompt_two,model,parser)
})

result = parallel_chain.invoke({'topic':'Struggling Life'})


print(result)