from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]
    
def get_retriever():
    urls = [
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    embedding_function = SentenceTransformerEmbedding('all-MiniLM-L6-v2')

    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding_function,
    )
    retriever = vectorstore.as_retriever(k=4)
    return retriever

def get_rag_chain():
    prompt = PromptTemplate(
        template="""Assume as an assistant for answering questions.
        Use the following documents to answer the questions.
        If you don't know the answer, just say that you don't know.
        Use at most four sentences and keep the answers concise:
        Question: {question}
        documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain