from langchain_classic.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

def make_critic_agent(indexes: dict[str, FAISS], llm):

    @tool
    def paper_critic(query: str)->str:
        """Critically evaluates each research paper for strengths, weaknesses, and novelty"""
        critiques=[]
        prompt = PromptTemplate(
            input_variables=["paper_id", "context"],
            template="""You are a critical peer reviewer. Evaluate this paper:
            
            Paper: {paper_id}
            Context: {context}

            Provide:
            1. Strengths
            2. Weakness (methodology gaps, missing baselines, limited scope, assumptions made)
            3. Novelty (is this a significant contribution)
            4. Verdict (Strong / Moderate / Weak) with one-line justification
            """
        )
        for paper_id, store in indexes.items():
            docs = store.similarity_search(
                "limitations future work methodology evaluation", k=5
            )
            context = "\n".join([d.page_content for d in docs])
            chain = prompt | llm
            result = chain.invoke({"paper_id": paper_id, "context": context})
            critiques.append(f"====Critique: {paper_id} ====\n {result.content}")
        return "\n\n".join(critiques)
    
    return paper_critic
        