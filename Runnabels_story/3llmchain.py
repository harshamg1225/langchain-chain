from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplates


# load the llm
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# create a prompt templates

template = PromptTemplates(
    input_variables=["topic"], template="suggest a catchy blog titlle about {topic}"
)

chain = LLMChain(llm=llm, prompt=template)

output = chain.run({"topic": "AI"})
print("generated blog title:", output)
