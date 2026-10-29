from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence,RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 1 

prompt_1 = PromptTemplate(
     template='Give me a one liner term about VC condition {condition}',
     input_variables=['condition']
)
prompt_2 = PromptTemplate(
     template='Give me a short concise explaination about the term {term}',
     input_variables=['term']
)

parser = StrOutputParser()


sequence_chain = RunnableSequence(prompt_1,model,parser)

chain_passthrough = RunnableParallel({
     "Vc_Topic": RunnablePassthrough(),
     "Explianed_Topic":RunnableSequence(prompt_2,model,parser)
})

final_chain = RunnableSequence(sequence_chain,chain_passthrough)

result = final_chain.invoke({"condition":'bankrupt'})

print(result)