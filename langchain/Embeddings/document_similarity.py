# There will be 5 document and will test the similarity of the user query
# Author: Muhammad Humayun Khan

from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large",dimensions = 300)

document = [
    "Umar Akmal is the worst crickerter",
    "Pakistani players are non-serious cricketeres",
    "Shaheen Shah Afridi is only t20 bowler at start of the career",
    "Haris Rauf is tapia bowler",
    "Babar Azam plays for own records nothing else",
    "Pakistani cricket will completely be like Zimbabwe in the coming five years"
]

user_query = "Tell me about the player in Pakistan cricket that plays for own records "

doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query(user_query)

# print(cosine_similarity([query_embeddings],doc_embeddings))  # print directly vectors
scores = cosine_similarity([query_embeddings],doc_embeddings)[0]

# the above result should be in the sorted order 
print(list(enumerate(scores)))  # there will be one index number attached to each score
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0] # sorted on the basis of second item means the similarity score will sort the list

print(user_query)
print(document[index])
print("Similarity index: ",score)
