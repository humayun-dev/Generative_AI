# Prompt Template is better than using the f string as Prompt Template having
# validation of the placeholder
# reusabilty
# Fit with the Langchain ecosystem

from langchain_core.prompts import PromptTemplate

# template for the prompt
template = PromptTemplate(
    template = '''
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable. 
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.
''', input_variables=['paper_input','style_input','length_input'],
validate_template = True
)

# now write the template in json format manually
with open("template.json", "w") as f:
    f.write(template.model_dump_json())
