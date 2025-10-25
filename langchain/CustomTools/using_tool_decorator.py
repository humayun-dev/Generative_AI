# Tools can be built-in and custom made as well in the LLM
# Tools are like the function or API that can be called by the LLM to perform a specific task which LLM can't
# It can be any specific business logic etc
# the tools can be build using @tool decorator, using structuredTool & pydantic, using BaseTool Class
# Author: Muhammad Humayun

from langchain_community.tools import tool

# using @tool decorator - create a function, hint for LLM, decorator
def multiply(a,b):
    """Multiply two numbers"""      # recommended doc string style - this is description of the tool
    return a * b

def multiply(a:int,b:int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def multiply(a:int,b:int) -> int:
    """Multiply two numbers"""
    return a * b

result = multiply.invoke({"a":3,"b":6})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)

