from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnableMap,
    RunnablePassthrough,
)
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a jocke aboyt the topic {topic}", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="write a summary about the jocke {joke}", input_variables=["joke"]
)
parser = StrOutputParser()

joke = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "summary_jocke": RunnableSequence(template2, model, parser),
    }
)

final_chain = RunnableSequence(joke, parallel_chain)

result = final_chain.invoke({"topic": "Cricket"})
print(result["summary_jocke"])
