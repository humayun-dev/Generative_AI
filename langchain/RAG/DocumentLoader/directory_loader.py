# Directory Loader in RAG is when we have to upload multiple files rather than only one
# Author: Muhammad Humayun Khan

from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader


# path = folder having different pdf files
# glob = pattern to select type of files
# loader_cls = PyPDFLoader as we are dealing with the pdf 
Loader = PyPDFDirectoryLoader(path="chapters",
                              glob="*.pdf",
                              loader_cls = PyPDFLoader)

document = Loader.load()

print(document[1].page_content)