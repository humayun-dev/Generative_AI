# pyPDFloader used in the RAG to read page by page the PDF document
# Author: Muhammad Humayun Khan

from langchain_community.document_loaders import PyPDFLoader

# create object of the pyPDFloader
Loader = PyPDFLoader('test.pdf')
docs = Loader.load()    # will get the document objects

print(docs)