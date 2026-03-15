from src.rag.document_loader import load_documents

def retrieve_context(source):
    knowledge = load_documents()
    return knowledge.get(source.lower(), "No knowledge available.")