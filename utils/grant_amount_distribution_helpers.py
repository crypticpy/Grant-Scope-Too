import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from typing import List, Dict, Any

from utils.utils import clean_label, format_currency, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data


def create_distribution_chart(df: pd.DataFrame, selected_clusters: List[str],
                              column_mapping: Dict[str, str]) -> go.Figure:
    """
    Create a bar chart showing the grant amount distribution by USD cluster.
    """
    filtered_df = df[df[column_mapping['Amount Usd Cluster']].isin(selected_clusters)]
    fig = px.bar(filtered_df,
                 x=column_mapping['Amount Usd Cluster'],
                 y=column_mapping['Amount Usd'],
                 color=column_mapping['Amount Usd Cluster'],
                 title="Grant Amount Distribution by USD Cluster")

    fig.update_layout(
        xaxis_title="USD Cluster",
        yaxis_title="Total Grant Amount (USD)",
        xaxis={'categoryorder': 'array', 'categoryarray': selected_clusters}
    )

    return fig


def create_cluster_comparison_chart(df: pd.DataFrame, selected_clusters: List[str],
                                    column_mapping: Dict[str, str]) -> go.Figure:
    """
    Create a stacked bar chart comparing grant counts and total amounts across clusters.
    """
    filtered_df = df[df[column_mapping['Amount Usd Cluster']].isin(selected_clusters)]
    cluster_stats = filtered_df.groupby(column_mapping['Amount Usd Cluster']).agg({
        column_mapping['Grant Key']: 'count',
        column_mapping['Amount Usd']: 'sum'
    }).reset_index()

    fig = go.Figure(data=[
        go.Bar(name='Grant Count', x=cluster_stats[column_mapping['Amount Usd Cluster']],
               y=cluster_stats[column_mapping['Grant Key']]),
        go.Bar(name='Total Amount (normalized)', x=cluster_stats[column_mapping['Amount Usd Cluster']],
               y=cluster_stats[column_mapping['Amount Usd']] / cluster_stats[column_mapping['Amount Usd']].max())
    ])

    fig.update_layout(
        title="Comparison of Grant Counts and Total Amounts Across Clusters",
        xaxis_title="USD Cluster",
        yaxis_title="Normalized Value",
        barmode='group'
    )

    return fig


def generate_distribution_analysis_prompt(df: pd.DataFrame, selected_clusters: List[str],
                                          column_mapping: Dict[str, str]) -> str:
    """
    Generate a prompt for AI analysis of the grant amount distribution.
    """
    total_grants = len(df)
    total_amount = df[column_mapping['Amount Usd']].sum()

    prompt = f"""
    Analyze the grant amount distribution for the following clusters: {', '.join(selected_clusters)}.

    Consider the following points in your analysis:
    1. The total number of grants ({total_grants}) and the total grant amount (${total_amount:,.2f}).
    2. The distribution of grants across the selected clusters.
    3. Any notable patterns or trends in the distribution.
    4. Potential implications of this distribution for grant seekers and funders.
    5. Comparison of grant counts vs total amounts in each cluster.
    6. Any outliers or unexpected findings in the data.

    Provide a comprehensive analysis with specific insights and data-driven observations.
    """
    return prompt


def ai_powered_distribution_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_clusters: List[str],
                                     selected_chart: str, selected_role: str, ai_enabled: bool, api_key: str,
                                     column_mapping: Dict[str, str], user_interactions: List[str]) -> str:
    """
    Perform AI-powered analysis on the grant amount distribution data.
    """
    if not ai_enabled or not api_key:
        return "AI analysis is not available. Please provide an API key to enable this feature."

    initial_prompt = generate_distribution_analysis_prompt(grouped_df, selected_clusters, column_mapping)
    additional_context = f"the distribution of grant amounts across different USD clusters: {', '.join(selected_clusters)}"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)
    dynamic_context = generate_dynamic_context(grouped_df, "Grant Amount Distribution", selected_chart,
                                               {"Selected Clusters": selected_clusters},
                                               user_interactions)

    full_prompt = pre_prompt + "\n" + dynamic_context + "\n" + initial_prompt

    try:
        analysis = query_data(grouped_df, full_prompt, dynamic_context, api_key)
        return analysis
    except Exception as e:
        return f"An error occurred during AI analysis: {str(e)}"