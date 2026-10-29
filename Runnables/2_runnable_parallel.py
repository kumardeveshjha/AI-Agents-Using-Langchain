from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

# Prompt 1 

# Prompt for LinkedIn Post
prompt_1 = PromptTemplate(
     template='Give me a Linkedin post about the topic {topic}',
     input_variables=['topic']
)

# Prompt for x Post
prompt_2 = PromptTemplate(
     template='Give me a X post about the topic {topic}',
     input_variables=['topic']
)


# Parser 

parser  = StrOutputParser()

# Parallelinzing of the chains 

parallel_chain = RunnableParallel({
     'LinkedIn_post':RunnableSequence(prompt_1,model,parser),
     'X_post':RunnableSequence(prompt_2,model,parser)
})

result = parallel_chain.invoke({'topic':'Product Market Fit'})

print(result['LinkedIn_post'])
