from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplates
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# load the documnet

loader = TextLoader("docs.txt")
document = loader.load()


# split the text into smaller chucks
text_splitter = RecursiveCharcterTextSplitter(chuck_size=500, chuck_overlap=50)

docs = text_splitter.split_documents(document)

# converts text innto embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# create a retrivers (fetch relevant document)
retriever = vectorstore.as_retriever()


llm = OpenAI()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "what are the key takweays frm the documnet?"

answer = qa_chain.run(query)
print("answers", answer)
