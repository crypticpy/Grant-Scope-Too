import streamlit as st
from typing import Callable, List
import time
from streamlit.components import v1 as components

def load_ai_analyses(analyses: List[Callable], analysis_names: List[str]):
    """
    Load AI analyses with progress indication and ensure page loads from top.

    :param analyses: List of functions that perform AI analyses
    :param analysis_names: List of names for each analysis (for display purposes)
    :return: List of analysis results
    """
    # Create a progress bar
    progress_bar = st.progress(0)

    # Create empty placeholders for each analysis
    placeholders = [st.empty() for _ in analyses]

    results = []
    for i, (analysis_func, name) in enumerate(zip(analyses, analysis_names)):
        # Update progress
        progress = (i + 1) / len(analyses)
        progress_bar.progress(progress)

        # Show loading message
        placeholders[i].info(f"Loading AI analysis for {name}...")

        # Perform analysis
        result = analysis_func()
        results.append(result)

        # Replace loading message with result
        placeholders[i].markdown(f"### AI Analysis: {name}")
        placeholders[i].markdown(result)

        # Small delay to show progress
        time.sleep(0.5)

    # Remove progress bar when done
    progress_bar.empty()


    # Ensure page scrolls to top
    js = "window.scrollTo(0, 0);"
    st.components.v1.html(f"<script>{js}</script>", height=0)

    return results