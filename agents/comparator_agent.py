from langchain_classic.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

def make_comparator_agent(indexes: dict[str, FAISS], llm):
    @tool
    def paper_comparator(query: str)->str:
        """Compares all uploaded research papers on a given topic. Input: a research topic or query"""
        
        excerpts_text=""

        for paper_id, store in indexes.items():
            docs = store.similarity_search(query=query, k=4)
            text = "\n".join([d.page_content for d in docs])
            excerpts_text += f"\n\n---{paper_id} ---\n{text}"
        prompt = PromptTemplate(
            input_variables=["query", "excerpts"],
            template="""
            You are a senior researcher. Compare these papers on: {query}
            {excerpts}
            Provide a structured comparison:
            1. Problem Statement
            2. Methodology
            3. Results
            4. Agreements
            5. Disagreements
            6. Which paper is stronger on this topic, and why?
            """
        )
        chain = prompt | llm
        result = chain.invoke({"query": query, "excerpts": excerpts_text})
        return result.content
    return paper_comparator