from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

model_1 = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# prompt 1 
prompt_1 = PromptTemplate(
     template='Give me a the short structured notes about the topic \n  {topic}',
     input_variables=['topic']
)
prompt_2 = PromptTemplate(
     template='Give me a 5 question quize about the \n {topic}',
     input_variables=['topic']
)
prompt_3 = PromptTemplate(
     template= """ Merge the given notes and buleted quiz in a single doc \n {notes} and \n  {quiz}
               
               """,
    input_variables=["notes", "quiz"],
)


# output parser 
parser = StrOutputParser()


# creating the chains for this model 

# parallel chain 

parallel_chain = RunnableParallel({
     'notes': prompt_1 | model_1 | parser,
     'quiz' : prompt_2 | model_1 |parser
})


# Sequence chain 

merge_chain = prompt_3 | model_1 | parser

chain = parallel_chain 


# Text 

text = "Blackhole Theory"

# Fnal result 

result = chain.invoke({'topic':text})

# print(result)

chain.get_graph().print_ascii()