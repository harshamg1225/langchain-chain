from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnableMap,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda,
)
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a detailed report about the {topic}", input_variables=["text"]
)

template2 = PromptTemplate(
    template="write a summary about the text {text}", input_variables=["text"]
)
parser = StrOutputParser()

chain1 = RunnableSequence(template1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(template2, model, parser)),
    RunnablePassthrough(),
)


final_chain = RunnableSequence(chain1, branch_chain)

result = final_chain.invoke({"topic": "russia vs ukarine"})

print(result)
