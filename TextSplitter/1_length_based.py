from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('GoogleMap.pdf')

doc = loader.load()

splitter = CharacterTextSplitter(
     chunk_size=100,
     chunk_overlap=0,
     separator=''
)


# text = "Hello hoi bye bye, why are yo uhere and why all this is happening in the current era of ai and and learning modes of engineering "
chunks = splitter.split_documents(doc)
print(chunks[0].metadata)