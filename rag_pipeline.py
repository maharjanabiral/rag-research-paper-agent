from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device":"cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def build_index_for_paper(pdf_path: str, paper_id: str) -> FAISS:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n', '\n\n', '.', '?']
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["paper_id"] = paper_id

    vector_store = FAISS.from_documents(chunks, EMBEDDING_MODEL)
    vector_store.save_local(f"faiss_index/{paper_id}")
    return vector_store

def load_all_indexes(paper_ids: list[str]) -> dict:
    return {
        pid: FAISS.load_local(
            f"faiss_index/{pid}",
            EMBEDDING_MODEL,
            allow_dangerous_deserialization=True
        )
        for pid in paper_ids
    }