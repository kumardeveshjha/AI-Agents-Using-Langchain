from langchain_community.retrievers import WikipediaRetriever


# query to retrieve from wikipedia 

query = "Tell us about Narendra Modi biography? "

wiki_retriever = WikipediaRetriever(top_k_results=1, lang="en")



docs = wiki_retriever.invoke(query)


print(docs[0])