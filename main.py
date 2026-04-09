import os
from rag_pipeline import build_index_for_paper, load_all_indexes, EMBEDDING_MODEL
from agents.supervisor import build_supervisor

PAPERS = {
    "attention_is_all_you_need": "papers/attention.pdf",
    "bert_paper": "papers/BERT.pdf"
}

CONFIG = {"configurable": {"thread_id": "research_session_1"}}

for paper_id, path in PAPERS.items():
    index_path = f"faiss_index/{paper_id}/index.faiss"
    if not os.path.exists(index_path):
        print(f"[*] Building index for: {paper_id}")
        build_index_for_paper(path, paper_id)
    else:
        print(f"[✓] Index already exists for: {paper_id}, skipping build.")

indexes = load_all_indexes(list(PAPERS.keys()))
agent = build_supervisor(indexes=indexes)

print("Research Paper Agent ready. \n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=CONFIG
    )
    print(f"\nAgent: {result['messages'][-1].content}\n")