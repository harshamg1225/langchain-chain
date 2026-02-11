from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# create a prompt template

prompt = PromptTemplate(
    input_variables=["topic"], templates="suggest a catchu blog tittle about {topic}"
)

# define the input

topic = input("enter a topic")

# format the prompt manually using prompttemplate
formatted_prompt = prompt.format(topic=topic)

# call the llm directly
blog_title = llm.predict(formatted_prompt)
print("gneerated blogs tittle", blog_title)
