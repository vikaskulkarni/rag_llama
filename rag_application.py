from doc_store import get_retriever, get_rag_chain

class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        documents = self.retriever.invoke(question)
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer

def main():
    retriever = get_retriever()
    rag_chain = get_rag_chain()
    
    rag_application = RAGApplication(retriever, rag_chain)
    
    question = "What is prompt engineering?"
    answer = rag_application.run(question)
    print("Question:", question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()