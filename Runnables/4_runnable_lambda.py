from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 1 

prompt = PromptTemplate(
     template='Give me a short explanantion of a VC topic {topic}',
     input_variables=['topic']
)

# Parser 

parser = StrOutputParser()

# Runnable Sequence to generate the VC topic

sequence_chain = RunnableSequence(prompt,model,parser)

# Parallel chain with lambda funation 

parallel_chain = RunnableParallel(
     {
          'topic_Explaine':RunnablePassthrough(),
          'word_count':RunnableLambda(lambda x: len(x.split()))
     }
)

final_chain = RunnableSequence(sequence_chain,parallel_chain)

print(final_chain.invoke({'topic':'sourcing'}))
