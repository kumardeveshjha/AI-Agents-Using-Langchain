# from langchain_google_genai import GoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()

# llm = GoogleGenerativeAI(model="gemini-3.5-flash-lite")

# result = llm.invoke("hi")

# print(result)



from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model='gemini-3.5-flash-lite')

print(llm.invoke("Hi I am Devesh Kumar"))




