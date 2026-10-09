"""A LangChain-compatible retrieval tool returning evidence and source IDs."""
from langchain_core.tools import tool
from rag import retrieve_policies


@tool
def search_university_it_policies(question: str) -> str:
    """Find relevant statements in the university IT policy knowledge base."""
    documents = retrieve_policies(question)
    if not documents:
        return 'NO_RELEVANT_POLICY_FOUND'
    return '\n\n'.join(
        f"[Source: {doc.metadata['policy_id']}; topic: {doc.metadata['topic']}]\n{doc.page_content}"
        for doc in documents
    )
