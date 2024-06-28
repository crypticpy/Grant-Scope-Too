import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
from typing import List, Union, Dict, Any

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, update_year_axis, generate_dynamic_context
from loaders.llama_index_setup import query_data

@handle_error
def create_scatter_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of grant description length vs. award amount.
    """
    fig = px.scatter(df, x='Description Word Count', y='Amount Usd', opacity=0.5,
                     title="Grant Description Length vs. Award Amount")
    fig.update_layout(
        xaxis_title='Number of Words in Grant Description',
        yaxis_title='Award Amount (USD)',
        width=800,
        height=600
    )
    return fig

@handle_error
def create_award_amount_chart(df: pd.DataFrame, selected_factor: str, chart_type: str) -> go.Figure:
    """
    Create a chart (bar or box) of average award amount by different factors.
    """
    exploded_df = df.explode(selected_factor)

    if chart_type == "Bar Chart":
        avg_amount_by_factor = exploded_df.groupby(selected_factor)['Amount Usd'].mean().reset_index()
        avg_amount_by_factor = avg_amount_by_factor.sort_values('Amount Usd', ascending=False)
        fig = px.bar(avg_amount_by_factor, x=selected_factor, y='Amount Usd',
                     title=f"Average Award Amount by {clean_label(selected_factor)}")
        fig.update_layout(
            xaxis_title=clean_label(selected_factor),
            yaxis_title='Average Award Amount (USD)',
            width=800,
            height=600,
            xaxis_tickangle=-45,
            xaxis_tickfont=dict(size=10)
        )
    else:  # Box Plot
        fig = px.box(exploded_df, x=selected_factor, y='Amount Usd',
                     title=f"Award Amount Distribution by {clean_label(selected_factor)}")
        fig.update_layout(
            xaxis_title=clean_label(selected_factor),
            yaxis_title='Award Amount (USD)',
            width=800,
            height=600,
            boxmode='group'
        )
    
    # Ensure that update_year_axis returns a valid Figure object
    try:
        fig = update_year_axis(fig, df, 'Year Issued')
    except Exception as e:
        st.warning(f"Error updating year axis: {str(e)}")

    return fig

@handle_error
def create_funder_affinity_chart(df: pd.DataFrame, selected_funder: str, selected_affinity_factor: str) -> go.Figure:
    """
    Create a bar chart showing funder affinity towards certain subjects, populations, or strategies.
    """
    funder_grants_df = df[df['Funder Name'] == selected_funder]
    exploded_funder_df = funder_grants_df.explode(selected_affinity_factor)
    funder_affinity = exploded_funder_df.groupby(selected_affinity_factor)['Amount Usd'].sum().reset_index()
    funder_affinity = funder_affinity.sort_values('Amount Usd', ascending=False)

    fig = px.bar(funder_affinity, x=selected_affinity_factor, y='Amount Usd',
                 title=f"Funder Affinity: {selected_funder} - {clean_label(selected_affinity_factor)}")
    fig.update_layout(
        xaxis_title=clean_label(selected_affinity_factor),
        yaxis_title='Total Award Amount (USD)',
        width=800,
        height=600,
        xaxis_tickangle=-45,
        xaxis_tickfont=dict(size=10)
    )
    return fig

@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for general analysis of relationships"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the general analysis of relationships. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "General Analysis of Relationships", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response

@handle_error
def general_analysis_relationships(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                                   ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("General Analysis of Relationships")
    st.write("""
    Welcome to the General Analysis of Relationships page! This section helps you uncover meaningful connections 
    and trends within the grant data. Explore relationships between various factors and award amounts to gain 
    valuable insights for your grant-related decisions, considering your specific project theme and goals.
    """)

    unique_grants_df = df.drop_duplicates(subset=['Grant Key'])
    unique_grants_df['Description Word Count'] = unique_grants_df['Grant Description'].apply(
        lambda x: len(str(x).split()))

    st.subheader("Relationship between Grant Description Length and Award Amount")
    try:
        fig = create_scatter_plot(unique_grants_df)
        st.plotly_chart(fig)

        if ai_enabled:
            scatter_analysis = generate_ai_analysis(
                unique_grants_df, grouped_df, selected_chart, selected_role, api_key,
                "scatter plot of grant description length vs. award amount",
                {}, custom_context
            )
            st.markdown(scatter_analysis)
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")

    st.subheader("Average Award Amount by Different Factors")
    factors = ['Grant Strategy Tran', 'Grant Population Tran', 'Grant Geo Area Tran', 'Funder Name']
    selected_factor = st.selectbox("Select Factor", options=factors)
    chart_type = st.radio("Select Chart Type", options=["Bar Chart", "Box Plot"])

    try:
        fig = create_award_amount_chart(unique_grants_df, selected_factor, chart_type)
        st.plotly_chart(fig)

        if ai_enabled:
            award_amount_analysis = generate_ai_analysis(
                unique_grants_df, grouped_df, selected_chart, selected_role, api_key,
                f"{chart_type.lower()} of award amount by {selected_factor}",
                {"selected_factor": selected_factor, "chart_type": chart_type}, custom_context
            )
            st.markdown(award_amount_analysis)
    except Exception as e:
        st.error(f"Error creating award amount chart: {str(e)}")

    st.subheader("Funder Affinity Analysis")
    funders = unique_grants_df['Funder Name'].unique().tolist()
    selected_funder = st.selectbox("Select Funder", options=funders)
    affinity_factors = ['Grant Subject Tran', 'Grant Population Tran', 'Grant Strategy Tran']
    selected_affinity_factor = st.selectbox("Select Affinity Factor", options=affinity_factors)

    try:
        fig = create_funder_affinity_chart(unique_grants_df, selected_funder, selected_affinity_factor)
        st.plotly_chart(fig)

        if ai_enabled:
            funder_affinity_analysis = generate_ai_analysis(
                unique_grants_df, grouped_df, selected_chart, selected_role, api_key,
                f"funder affinity chart for {selected_funder} and {selected_affinity_factor}",
                {"selected_funder": selected_funder, "selected_affinity_factor": selected_affinity_factor}, custom_context
            )
            st.markdown(funder_affinity_analysis)
    except Exception as e:
        st.error(f"Error creating funder affinity chart: {str(e)}")

    if ai_enabled:
        st.subheader("AI-Assisted Analysis")
        additional_context = "relationships between grant description length, award amounts, and various factors such as strategies, populations, and geographical areas"
        pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

        query_options = [
            "What insights can we draw from the relationship between grant description length and award amount?",
            f"What are the key trends in award amounts for different {selected_factor}s?",
            f"How does the funder {selected_funder} prioritize different {selected_affinity_factor}s in their funding?",
            "Are there any notable outliers or patterns in the data that warrant further investigation?",
            "How do different factors (strategies, populations, geo areas) compare in terms of their impact on award amounts?",
            "Custom Question"
        ]

        selected_query = st.selectbox("Select a predefined question or choose 'Custom Question':", query_options)

        if selected_query == "Custom Question":
            user_query = st.text_input("Enter your question here:")
            query_text = user_query
        else:
            query_text = selected_query

        if st.button("Submit", key="ai_analysis_submit"):
            if query_text:
                response = generate_ai_analysis(
                    df, grouped_df, selected_chart, selected_role, api_key,
                    "user question",
                    {"query": query_text}, custom_context, query_text
                )
                st.markdown(response)
            else:
                st.warning("Please enter a question or select a predefined question.")
    else:
        st.info("AI-assisted analysis is disabled. Please provide an API key to enable this feature.")

    if st.checkbox("Show Underlying Data", key="show_underlying_data01"):
        st.write(unique_grants_df)

    if st.button("Download Data as CSV"):
        csv_data = unique_grants_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="grant_data.csv",
            mime="text/csv"
        )

    st.markdown("""
    This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods 
    for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. 
    Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
    """)
