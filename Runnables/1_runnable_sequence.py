from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 1 

prompt_1 = PromptTemplate(
     template='Give me a detailed explanantion of a VC topic {topic}',
     input_variables=['topic']
)
prompt_2 = PromptTemplate(
     template='Give me 5 key points about this topic : {topic}',
     input_variables=['topic']
)

parser = StrOutputParser()


sequence_chain = RunnableSequence(prompt_1,model,parser,prompt_2,model,parser)

result = sequence_chain.invoke({'topic':'aquisition'})

print(result)


