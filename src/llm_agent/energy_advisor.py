from src.rag.retriever import retrieve_context

def explain_energy(source):

    context = retrieve_context(source)

    explanation = f"""
    Recommended energy source: {source}

    Explanation:
    {context}
    """

    return explanation