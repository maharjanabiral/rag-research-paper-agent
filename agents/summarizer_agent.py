from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.vectorstores import FAISS

def make_summarizer_tool(indexes: dict[str, FAISS], llm):

    @tool
    def paper_summarizer(query: str)->str:
        """Generates a structured summary for each uploaded research paper"""
        prompt = PromptTemplate(
            input_variables=["paper_id", "context"],
            template=""""You are a research assistant. Summarize the following paper
            Paper ID : {paper_id}
            Context : {context}

            Provide:
            1. Main Objective
            2. Methodology(1-2 sentences)
            3. Key results (bullet points)
            4. Conclusion
            """
        )
        chain = prompt | llm
        summaries = []
        for paper_id, store in indexes.items():
            docs = store.similarity_search(
                "research objective methodology results conclusion", k=6
            )
            context = "\n\n".join([d.page_content for d in docs])
            result = chain.invoke({"paper_id": paper_id, "context": context})
            summaries.append(f"=== {paper_id} ===\n{result['text']}")  
        return '\n\n'.join(summaries)
    
    return paper_summarizer