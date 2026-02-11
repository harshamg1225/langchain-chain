from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnableMap,
    RunnablePassthrough,
    RunnableLambda,
)
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a joke about the topic {topic}", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="write a summary about the jocke {joke}", input_variables=["joke"]
)
parser = StrOutputParser()

joke = RunnableSequence(template1, model, parser)


def word_count(x):
    return len(x.split())


parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count),
    }
)

full_chain = RunnableSequence(joke, parallel_chain)

result = full_chain.invoke({"topic": "cricket"})

print(result)
