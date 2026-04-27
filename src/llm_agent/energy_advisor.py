from src.rag.retriever import retrieve_context
import logging

logger = logging.getLogger(__name__)


def explain_energy(source):
    """
    Generate explanation for energy recommendation.
    
    Args:
        source (str): Recommended energy source ("Solar" or "Wind")
    
    Returns:
        str: Explanation text
    """
    try:
        # Validate source type
        if not isinstance(source, str):
            logger.warning(f"Invalid source type: {type(source).__name__}, expected str")
            return "Unable to generate explanation: Invalid source format."
        
        # Retrieve knowledge context
        context = retrieve_context(source)
        
        if context == "No knowledge available.":
            logger.warning(f"No knowledge available for source: {source}")
        
        explanation = f"""Recommended energy source: {source}

Explanation:
{context}"""
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation for source '{source}': {e}", exc_info=True)
        return f"Unable to generate explanation for {source} at this time."