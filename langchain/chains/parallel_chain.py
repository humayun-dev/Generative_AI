# Parallel Chain or Linear Chain
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from pathlib import Path    # to read text from another file for the parallel chain

# load the key and the model
load_dotenv()
model_one = ChatOpenAI()
model_two = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# two prompts - one for the notes and another for the quiz
prompt = PromptTemplate(
    template = 'Generate simple notes from the following text \n {text}',
    input_variables = ['text']
)
prompt_two = PromptTemplate(
    template = 'Generate 5 short question answers from the following text \n {text}',
    input_variables = ['text']
)

# prompt to merge the quiz and notes
prompt_three = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n {notes_chain} and {quiz_chain}',
    input_variables = ['notes_chain','quiz_chain']
)

parser = StrOutputParser()

# Now create the parallel chain through runnables. The chains will be of the notes and the quiz
parallel_chains = RunnableParallel({
    'notes_chain': prompt | model_one | parser,
    'quiz_chain': prompt_two | model_two | parser
})

# Now create chain to merge the notes and quiz chain
merge_chain = prompt_three | model_one | parser

# Pipeline the Merged all the chains
chain = parallel_chains | merge_chain

# read the text for the execution process from the file text_template.txt
prompt_path = Path("text_template.txt")
text = prompt_path.read_text(encoding="utf-8")

result = chain.invoke({'text':text})    # the first step to start execution
print(result)

# visualize in graph
chain.get_graph().print_ascii()

