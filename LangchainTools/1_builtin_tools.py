# This is the basic duckduck go tool for langchain 
# Built in tools are the tools that are already provide by the langchain community 

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import ShellTool
from sklearn.utils import resample

search_tool = DuckDuckGoSearchRun()
parser = StrOutputParser()
# result = search_tool.invoke("What is the price of a stock of jio ?")

# print(parser.parse(result))


# Shell tool

shell = ShellTool()

res = shell.invoke('directoryname')

print(res)