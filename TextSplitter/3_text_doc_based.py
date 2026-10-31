from langchain_classic.text_splitter import RecursiveCharacterTextSplitter,PythonCodeTextSplitter
from langchain_text_splitters import Language

splitter = PythonCodeTextSplitter(
     chunk_size=500,
     chunk_overlap=20
     
)

code = """

               class SimpleTextSplitter:
          def __init__(self, separator="\n\n", chunk_size=100, chunk_overlap=20):
               "
               Initializes the text splitter with behavior similar to LangChain.
               "
               self.separator = separator
               self.chunk_size = chunk_size
               self.chunk_overlap = chunk_overlap

          def split_text(self, text: str) -> list:
               "
               Splits a string into chunks based on the separator, size, and overlap.
               "
               # Step 1: Split the text into raw paragraphs/sections by the separator
               splits = text.split(self.separator)
               
               chunks = []
               current_chunk = ""

               for split in splits:
                    # If adding this split exceeds the chunk size, save the current chunk
                    if current_chunk and (len(current_chunk) + len(self.separator) + len(split) > self.chunk_size):
                         chunks.append(current_chunk)
                         
                         # Create the overlap text using the end of the previous chunk
                         overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                         current_chunk = current_chunk[overlap_start:]
                         
                         # Re-add the separator if we are carrying text over
                         if current_chunk:
                              current_chunk += self.separator

                    # Append the current split piece to the active chunk
                    if current_chunk:
                         current_chunk += split
                    else:
                         current_chunk = split

               # Append the final remaining chunk if it's not empty
               if current_chunk:
                    chunks.append(current_chunk)

               return chunks


          # ==========================================
          # Example Usage (Creating the Objects)
          # ==========================================

          # 1. Define sample text
          essay = (
          "Artificial intelligence is transforming modern technology."
          "\n\n"
          "Large Language Models require text chunking to fit context windows."
          "\n\n"
          "Simple splitters make this process easy without external libraries."
          "\n\n"
          "Vector databases then store these chunks as embeddings."
          )

          # 2. Instantiate the splitter object with specific parameters
          my_splitter = SimpleTextSplitter(
          separator="\n\n", 
          chunk_size=120, 
          chunk_overlap=25
          )

          # 3. Process the text into text chunks
          resulting_chunks = my_splitter.split_text(essay)

          # 4. Print out the results to inspect the boundaries
          print(f"Total Chunks Created: {len(resulting_chunks)}\n")

          for i, chunk in enumerate(resulting_chunks):
          print(f"--- Chunk {i + 1} ({len(chunk)} characters) ---")
          print(chunk)
          print()

       
"""

chunk = splitter.split_text(code)

print(chunk[1])