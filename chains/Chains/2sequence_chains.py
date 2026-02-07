from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write detailed report on {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write summary based on the given {text}", input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser


result = chain.invoke({"topic": "runnable in langchain"})

print(result)
