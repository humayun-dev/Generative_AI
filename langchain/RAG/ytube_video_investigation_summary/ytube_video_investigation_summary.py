
# Project - Sometimes the videos on the youtube is very long and we can't figure out either my topic of interest exisits or not
# In this system, the video will be analyzed/investigate for the topic interest and after that it can summarize the video as well
# first the manaul architecture of RAG is implemented then the runnable chaining is applied
# Author: Muhammad Humayun Khan

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# load key and the model
load_dotenv()
model = ChatOpenAI()

# For RAG application, there are four steps such as indexing, retrieving, augmentation and generation

# Step 1: Indexing - fetch the data source
youtube_video_id = "ppaK_PcOAAI"

# Fetch the transcript and convert to plain text
try:
    ytube_api = YouTubeTranscriptApi()
    transcript = ytube_api.fetch(youtube_video_id)

    # Join all transcript pieces into one full string
    transcript = "\n".join([item.text for item in transcript.snippets])

    print("Transcript fetched successfully!\n")
    print(transcript)  

except Exception as e:
    print("Transcript not available:", e)
    transcript = ""

# Indexing - Data Source into chuncks via text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
chunks = splitter.create_documents([transcript])
print("The number of chunks are; ",len(chunks))

# Indexing - Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks,embeddings)

# Step 2 - Retreiver
retreiver = vector_store.as_retriever(search_type = "similarity",search_kwarg={'k':4})
# retreiver.invoke("What is Discipline")

# Step 3 - Augmentation - merging retriver generated relevant docs and query to create a prompt
prompt = PromptTemplate(
    template = """
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question:{question}
""",
input_variables=['context','question']
)

# question = "is there any kind of village name or boy name mentioned?"
# retrieved_docs = retreiver.invoke(question)

# as it will create 4 docs so join its page content
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# merge the context and question
#final_prompt = prompt.invoke({"context":context_text,"question":question})


# Step 4 - Generation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
#response = llm.invoke(final_prompt)
#print(response.content)

# building chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retreiver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke("is there any kind of village name or boy name mentioned?")

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke("Can you summarize the video?"))


