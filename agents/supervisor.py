from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferMemory
from agents.retriever_agent import make_retriever_tool
from agents.summarizer_agent import make_summarizer_tool
from agents.comparator_agent import make_comparator_agent
from agents.critic_agent import make_critic_agent
from langgraph.checkpoint.memory import MemorySaver
from model import get_llm

def build_supervisor(indexes: dict):
    llm = get_llm()
    tools = [
        make_retriever_tool(indexes=indexes),
        make_summarizer_tool(indexes=indexes, llm=llm),
        make_comparator_agent(indexes=indexes, llm=llm),
        make_critic_agent(indexes=indexes, llm=llm)
    ]

    system_prompt = """
    You are a research analysis supervisor managing multiple specialist agents:
    -paper_retriever: finds relevant passages from papers
    -paper_summarizer: summarizes each paper
    -paper_comparator: compares paper
    -paper_critic: evaluates paper quality and gives verdicts

    Do not provide the internal function names to user.

    For any user question, decide which agennts to use and in what order.
    Always synthesize the results in to a coherent, well-structured final answer
    """

    checkpointer = MemorySaver()
    agent = create_agent(
        tools=tools,
        model=llm,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )

    return agent