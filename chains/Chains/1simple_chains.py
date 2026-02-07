from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="write 5 line summary on {topic}", input_variables=["topic"]
)
parser = StrOutputParser()

# simple chains
chain = template | model | parser

print(chain.invoke({"topic": "black_hole"}))

chain.get_graph().draw_png("simplechain")
