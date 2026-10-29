from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnableSequence,RunnablePassthrough,RunnableParallel,RunnableLambda,RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 1 

prompt_1 = PromptTemplate(
     template='Give me a Report of a VC topic {topic}',
     input_variables=['topic']
)


prompt_2 = PromptTemplate(
     template='Give me the summary of this Report {report}',
     input_variables=['report']
)

parser = StrOutputParser()

# sequence Chain : Topic Report 

sequence_chain = RunnableSequence(prompt_1,model,parser)

# Creating a branch consist of the full flow of the chain 

branch_chain = RunnableBranch(
     (lambda x:len(x.split())>500, RunnableSequence(prompt_2,model,parser)),
     (RunnablePassthrough())
) 

chain = RunnableSequence(sequence_chain,branch_chain)

print(chain.invoke({'topic':"Due-Dilligence"}))