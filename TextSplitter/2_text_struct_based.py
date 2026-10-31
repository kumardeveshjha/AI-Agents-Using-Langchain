from langchain_classic.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
     chunk_size=30,
     chunk_overlap=10
)


text = """
        Also, I wanted to ask you something about your own experience. I'm quite interested in AI Product Engineering because it combines building AI systems with product thinking, which aligns with my interests.
        Since you're working in this space, what skills or experiences would you recommend focusing on to move towards an AI Product Engineer role? I'd really value your perspective, especially as someone who's already making that transition
"""
chunk = splitter.split_text(text)

print(chunk)