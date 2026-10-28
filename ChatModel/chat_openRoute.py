from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(
     model="nvidia/nemotron-3.5-super-vl:free",
)

query= str(input("""Enter your query here:"""))

result = model.invoke(query)

print(result.content)
