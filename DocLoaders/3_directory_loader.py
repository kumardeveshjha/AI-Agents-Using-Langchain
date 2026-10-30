# this takes time to load the docs and not appropriate when we have large number of pdf with size and content

from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
     path='Book',
     glob='*.pdf*',
     loader_cls=PyPDFLoader
)

# doc = loader.load() # this ibest when we are dealing with a small pdf and we want the pdf to be opened upfront 

# When we have a large number of pdf and wanted to process fast and don't want it to load in the memory (RAM)

docs = loader.lazy_load()


for doc in docs:
     print(doc.metadata)