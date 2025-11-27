"""RAG chain creation for baseline and advanced systems."""
import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.retrievers import BaseRetriever


def create_llm(config: dict):
    """
    Create LLM instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLM instance
    """
    llm_provider = config.get("llm_provider", "azure_openai")
    
    if llm_provider == "azure_openai":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = config.get("azure_deployment_name") or config.get("fast_llm")
        api_version = config.get("azure_api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        
        return AzureChatOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            temperature=0
        )
    else:
        return ChatOpenAI(model=config.get("fast_llm", "gpt-3.5-turbo"), temperature=0)


def create_baseline_rag_chain(
    retriever: BaseRetriever,
    config: dict,
    system_prompt: Optional[str] = None
):
    """
    Create a baseline RAG chain.
    
    Args:
        retriever: Document retriever
        config: Configuration dictionary
        system_prompt: Optional custom system prompt
        
    Returns:
        RAG chain
    """
    if system_prompt is None:
        system_prompt = """You are an AI energy sector analyst specializing in renewable energy, green hydrogen, and energy transition. Answer the question based only on the following context from energy sector documents:

{context}

Question: {question}

Provide a clear, accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
    
    template = system_prompt
    prompt = ChatPromptTemplate.from_template(template)
    llm = create_llm(config)
    
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

