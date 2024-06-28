import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from typing import Dict, Callable, Any
from utils.data_summary_helpers import (
    create_summary_metrics,
    create_top_funders_chart,
    create_funder_type_pie_chart,
    create_subject_area_chart,
    create_time_series_chart,
    create_interactive_network_graph,
    format_large_number
)

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context, \
    generate_column_mapping
from loaders.llama_index_setup import query_data
from utils.utils import convert_streamlit_to_pdf_buffer


def load_individual_analysis(analysis_func: Callable, chart_number: int, total_charts: int) -> str:
    with st.spinner(f"Chart {chart_number} of {total_charts} analysis is loading... Please wait."):
        return analysis_func()


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for data summary"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    query = f"Analyze the {analysis_type} for the data summary. Provide insights, trends, and implications."
    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Data Summary", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def ai_powered_data_summary(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                            ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.title("GrantScope AI-Powered Dashboard")
    st.write("""
    Welcome to the AI-Enhanced Data Summary page! This interactive dashboard provides a comprehensive overview of 
    your grant dataset, offering valuable insights into funding patterns, key players, and trends in the philanthropic 
    sector. Our AI assistant will guide you through the analysis, highlighting key points and suggesting areas for 
    further exploration.
    """)

    if not ai_enabled:
        st.warning("AI analysis is currently disabled. Please provide an API key to enable AI-powered insights.")
        return

    # Generate column mapping
    column_mapping = generate_column_mapping(df)

    # Initialize session state for user interactions
    if 'user_interactions' not in st.session_state:
        st.session_state.user_interactions = []

    # Dataset Overview
    st.header("Dataset Overview")
    create_summary_metrics(df, column_mapping)

    def analyze_overview():
        overview_prompt = "Provide an overview of the grant dataset that a grant writer or program manager seeking funding will find helpful. Highlighting key metrics and any notable characteristics."
        context = generate_dynamic_context(df, "Data Summary", selected_chart, {},
                                           st.session_state.user_interactions,
                                           custom_context if isinstance(custom_context, dict) else None)
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key, "dataset overview", {}, custom_context)

    overview_analysis = load_individual_analysis(analyze_overview, 1, 6)
    st.markdown(overview_analysis)

    with st.expander("View Data Sample"):
        st.write(df.head())

    # Funding Landscape
    st.header("Funding Landscape")

    # Top Funders Analysis
    st.subheader("Top Funders by Total Grant Amount")
    top_n = st.slider("Select the number of top funders to display", min_value=5, max_value=20, value=10, step=1)
    fig = create_top_funders_chart(df, top_n, column_mapping)
    st.plotly_chart(fig, use_container_width=True)

    def analyze_top_funders():
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    f"top {top_n} funders analysis", {"top_n": top_n}, custom_context)

    top_funders_analysis = load_individual_analysis(analyze_top_funders, 2, 6)
    st.markdown(top_funders_analysis)

    # Funder Type Distribution
    st.subheader("Grant Distribution by Funder Type")
    fig = create_funder_type_pie_chart(df, column_mapping)
    st.plotly_chart(fig, use_container_width=True)

    def analyze_funder_type():
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    "funder type distribution analysis", {}, custom_context)

    funder_type_analysis = load_individual_analysis(analyze_funder_type, 3, 6)
    st.markdown(funder_type_analysis)

    # Grant Characteristics
    st.header("Grant Characteristics")

    # Subject Area Analysis
    st.subheader("Grant Distribution by Subject Area")
    fig = create_subject_area_chart(df, column_mapping)
    st.plotly_chart(fig, use_container_width=True)

    def analyze_subject_area():
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    "subject area distribution analysis", {}, custom_context)

    subject_area_analysis = load_individual_analysis(analyze_subject_area, 4, 6)
    st.markdown(subject_area_analysis)

    # Time Series Analysis
    st.subheader("Total Grant Amount Over Time")
    fig = create_time_series_chart(df, column_mapping)
    st.plotly_chart(fig, use_container_width=True)

    def analyze_time_series():
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    "time series analysis", {}, custom_context)

    time_series_analysis = load_individual_analysis(analyze_time_series, 5, 6)
    st.markdown(time_series_analysis)

    # Network Analysis
    st.subheader("Interactive Network of Funders and Recipients")
    fig = create_interactive_network_graph(df, column_mapping)
    st.plotly_chart(fig, use_container_width=True)

    def analyze_network():
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    "network analysis", {}, custom_context)

    network_analysis = load_individual_analysis(analyze_network, 6, 6)
    st.markdown(network_analysis)

    # Comprehensive AI Analysis
    st.header("AI-Generated Comprehensive Analysis")

    def comprehensive_analysis():
        comprehensive_prompt = """
        Provide a comprehensive analysis of the entire dataset based on all the visualizations and insights generated. 
        Include the following:
        1. Overall summary of the funding landscape
        2. Key trends and patterns observed
        3. Potential opportunities for grant seekers
        4. Areas that may require further investigation
        5. Recommendations for different types of users (e.g., grant writers, researchers, nonprofit organizations)
        """
        if isinstance(custom_context, dict) and 'project_theme' in custom_context:
            comprehensive_prompt += f"\nConsider the project theme: {custom_context['project_theme']}"
        
        return generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                    "comprehensive analysis", {}, custom_context)

    with st.spinner("Generating comprehensive analysis... This may take a moment."):
        comprehensive_analysis_result = comprehensive_analysis()
    st.markdown(comprehensive_analysis_result)

    # User Interaction and Feedback
    st.header("Explore Further")
    user_question = st.text_input("Ask a question about the data or request further analysis:")
    if user_question:
        st.session_state.user_interactions.append(f"User asked: {user_question}")
        with st.spinner("Analyzing your question..."):
            ai_response = generate_ai_analysis(df, grouped_df, selected_chart, selected_role, api_key,
                                               "user question", {}, custom_context)
        st.markdown(ai_response)

    st.markdown("---")
    st.subheader("Download Full Report")

    if 'pdf_generated' not in st.session_state:
        st.session_state.pdf_generated = False

    if st.button("Generate PDF Report", key="generate_pdf_report_button1"):
        with st.spinner("Generating PDF..."):
            html_content = st._get_docstring()
            pdf_buffer = convert_streamlit_to_pdf_buffer(html_content)
            st.session_state.pdf_buffer = pdf_buffer
            st.session_state.pdf_generated = True

    if st.session_state.pdf_generated:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state.pdf_buffer,
            file_name="grant_analysis_report.pdf",
            mime="application/pdf"
        )

    st.markdown("""
    This AI-powered dashboard was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using state-of-the-art AI and data visualization 
    techniques. It leverages the Candid API, Streamlit, Plotly, and other open-source libraries. The AI analyses are 
    powered by advanced AI language models to provide dynamic, context-aware insights.
    """)
    