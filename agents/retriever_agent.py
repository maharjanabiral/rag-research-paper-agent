from langchain.tools import tool
from langchain_community.vectorstores import FAISS

def make_retriever_tool(indexes: dict):
    @tool
    def paper_retriever(query: str) -> str:
        """Searches all uploaded research papers for relevant passages. Input : a question or topic"""
        for paper_id, store in indexes.items():
            docs = store.similarity_search(query, k=3)
            all_results = [f"[{paper_id}] {doc.page_content}" for doc in docs]
        
        return "\n\n----\n\n".join(all_results)
    return paper_retriever
