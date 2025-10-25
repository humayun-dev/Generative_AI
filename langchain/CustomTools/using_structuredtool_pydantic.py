# Tools can be built-in and custom made as well in the LLM
# Tools are like the function or API that can be called by the LLM to perform a specific task which LLM can't
# It can be any specific business logic etc
# the tools can be build using @tool decorator, using structuredTool & pydantic, using BaseTool Class
# Author: Muhammad Humayun

from langchain_community.tools import StructuredTool
from pydantic import BaseModel,Field

# create a pydantic schema class
class Multiply_class(BaseModel):
    a:int = Field(...,description = "First number to multiply")
    b:int = Field(..., description = "Second number to multiply")

# create a function
def multiply_func(a:int,b:int) -> int:
    """Multiply two numbers"""
    return a * b

# create structedTool function
custom_structured_tool = StructuredTool.from_function(
    func = multiply_func,
    name = "multiply",
    description = "Multiply two numbers",
    args_schema = Multiply_class
)

result = custom_structured_tool.invoke({"a":3,"b":6})

print(result)
print(custom_structured_tool.name)
print(custom_structured_tool.description)
print(custom_structured_tool.args)


