import openai
from typing import Optional
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
import pandas as pd
import streamlit as st

@st.cache_resource
def setup_llama_index(api_key: str) -> None:
    """
    Set up LlamaIndex with specific model and settings.

    Args:
        api_key (str): The OpenAI API key.
    """
    Settings.llm = OpenAI(api_key=api_key, temperature=0.2, model="gpt-4o")

def initialize_openai(api_key: str) -> None:
    """
    Initialize OpenAI API with the API key.

    Args:
        api_key (str): The OpenAI API key.
    """
    openai.api_key = api_key

def query_data(df: pd.DataFrame, query_text: str, static_pre_prompt: str, dynamic_context: str, api_key: str) -> Optional[str]:
    """
    Query data using LlamaIndex.

    Args:
        df (pd.DataFrame): The DataFrame to query.
        query_text (str): The user's query text.
        static_pre_prompt (str): The static pre-prompt to provide context for the query.
        dynamic_context (str): The dynamic context based on current user interactions and data view.
        api_key (str): The OpenAI API key.

    Returns:
        Optional[str]: The response from the query engine, or None if an error occurs.
    """
    setup_llama_index(api_key)
    initialize_openai(api_key)

    full_prompt = f"{static_pre_prompt}\n\nCurrent Context:\n{dynamic_context}\n\nUser Query: {query_text}"

    query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
    response = query_engine.query(full_prompt)

    if response and response.response:
        cleaned_response = response.response.replace("$", "\$")
        return cleaned_response
    else:
        st.warning("No response generated from the query engine.")
        return None

def ai_analysis(df: pd.DataFrame, query: str, static_pre_prompt: str, dynamic_context: str, api_key: str) -> Optional[str]:
    """
    Perform AI analysis on the data.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        query (str): The user's query or analysis request.
        static_pre_prompt (str): The static pre-prompt to provide context for the analysis.
        dynamic_context (str): The dynamic context based on current user interactions and data view.
        api_key (str): The OpenAI API key.

    Returns:
        Optional[str]: The AI-generated analysis, or None if an error occurs.
    """
    try:
        setup_llama_index(api_key)
        initialize_openai(api_key)

        full_prompt = f"""
        Static Context:
        {static_pre_prompt}

        Dynamic Context:
        {dynamic_context}

        Query: {query}

        Please provide a detailed analysis based on the given context and query. 
        Include relevant statistics and insights from the data.
        """

        query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
        response = query_engine.query(full_prompt)

        if response and response.response:
            return response.response
        else:
            st.warning("No analysis generated from the AI.")
            return None
    except Exception as e:
        st.error(f"An error occurred during AI analysis: {str(e)}")
        return None
