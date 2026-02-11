from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharcterTextSplitter


# load the document
loader = TextLoader("docs.txt")
document = loader.load()


# split the text into smaller chucks
text_splitter = RecursiveCharcterTextSplitter(chuck_size=500, chuck_overlap=50)

docs = text_splitter.split_documents(document)

# converts text innto embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# create a retrivers (fetch relevant document)
retriever = vectorstore.as_retriever()

# manually retriever relabeny documnet
query = "what are the key takeways from the document?"
retrived_docs = retriever.get_relevant_document(query)

# combine retrived text into single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrived_docs])

# initialize the llms

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

retrieved_text = "\n".join([doc.page_content for doc in retrived_docs])
prompt = (
    f"based on the following text, answer the question: {query}\n\n {retrieved_text}"
)
answer = llm.predict(prompt)

print(answer)
