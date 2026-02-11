from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)
template1 = PromptTemplate(template="write joke on {topic}", input_variables=["topic"])
parser = StrOutputParser()
template2 = PromptTemplate(
    template="explain the following joke {joke}", input_variables=["joke"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)

result = chain.invoke({"topic": "cricket"})
print(result)
