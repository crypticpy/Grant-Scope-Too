import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from typing import List, Dict, Any

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data


@handle_error
def create_distribution_chart(df: pd.DataFrame, selected_clusters: List[str]) -> go.Figure:
    filtered_df = df[df['Amount Usd Cluster'].isin(selected_clusters)]
    cluster_totals = filtered_df.groupby('Amount Usd Cluster')['Amount Usd'].sum().reset_index()

    fig = px.bar(cluster_totals,
                 x='Amount Usd Cluster',
                 y='Amount Usd',
                 color='Amount Usd Cluster',
                 labels={'Amount Usd': 'Total Grant Amount (USD)', 'Amount Usd Cluster': 'USD Cluster'},
                 title="Grant Amount Distribution by USD Cluster")

    fig.update_layout(
        xaxis_title="USD Cluster",
        yaxis_title="Total Grant Amount (USD)",
        xaxis={'categoryorder': 'array', 'categoryarray': selected_clusters},
        showlegend=False
    )

    fig.update_traces(
        texttemplate='$%{y:,.0f}',
        textposition='outside',
        textangle=0,
        textfont=dict(size=10)
    )

    return fig


@handle_error
def create_cluster_comparison_chart(df: pd.DataFrame, selected_clusters: List[str]) -> go.Figure:
    filtered_df = df[df['Amount Usd Cluster'].isin(selected_clusters)]
    cluster_stats = filtered_df.groupby('Amount Usd Cluster').agg({
        'Grant Key': 'count',
        'Amount Usd': 'sum'
    }).reset_index()

    max_count = cluster_stats['Grant Key'].max()
    max_amount = cluster_stats['Amount Usd'].max()
    cluster_stats['Normalized Amount'] = cluster_stats['Amount Usd'] / max_amount * max_count

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cluster_stats['Amount Usd Cluster'],
        y=cluster_stats['Grant Key'],
        name='Grant Count',
        text=cluster_stats['Grant Key'],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=cluster_stats['Amount Usd Cluster'],
        y=cluster_stats['Normalized Amount'],
        name='Total Amount (Normalized)',
        text=cluster_stats['Amount Usd'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside'
    ))

    fig.update_layout(
        title="Comparison of Grant Counts and Total Amounts Across Clusters",
        xaxis_title="USD Cluster",
        yaxis_title="Count / Normalized Amount",
        barmode='group',
        legend_title_text='Metric'
    )

    return fig

@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    additional_context = f"the {analysis_type} for grant amount distribution across USD clusters: {', '.join(context.get('Selected Clusters', []))}"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the selected USD clusters. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Grant Amount Distribution", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df[df['Amount Usd Cluster'].isin(context.get('Selected Clusters', []))],
                          query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def grant_amount_distribution(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                              ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Powered Grant Amount Distribution Analysis")
    st.write("""
    Welcome to the AI-Enhanced Grant Amount Distribution Analysis page! This interactive tool allows you to explore
    how grants are distributed across various USD clusters. Our AI assistant will provide insights and analysis
    based on the visualizations and data presented, considering your project theme and goals.
    """)

    cluster_options = sorted(grouped_df['Amount Usd Cluster'].unique().tolist())
    selected_clusters = st.multiselect("Select USD Clusters", options=cluster_options, default=cluster_options)

    if selected_clusters:
        st.subheader("Grant Amount Distribution by USD Cluster")
        fig_distribution = create_distribution_chart(grouped_df, selected_clusters)
        st.plotly_chart(fig_distribution)

        if ai_enabled:
            distribution_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "distribution chart", {"Selected Clusters": selected_clusters}, custom_context
            )
            st.markdown(distribution_analysis)

        st.subheader("Comparison of Grant Counts and Total Amounts Across Clusters")
        fig_comparison = create_cluster_comparison_chart(grouped_df, selected_clusters)
        st.plotly_chart(fig_comparison)

        if ai_enabled:
            comparison_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "cluster comparison chart", {"Selected Clusters": selected_clusters}, custom_context
            )
            st.markdown(comparison_analysis)

        if ai_enabled:
            st.subheader("AI-Generated Comprehensive Analysis")
            comprehensive_prompt = f"""
            Provide a comprehensive analysis of the grant amount distribution across the selected USD clusters 
            ({', '.join(selected_clusters)}). Include the following:
            1. Overall trends and patterns in the distribution
            2. Notable differences between clusters in terms of grant count and total amount
            3. Potential implications for grant seekers and funders
            4. Any anomalies or interesting findings
            5. Recommendations for further investigation or analysis
            6. How this distribution relates to the project theme and goals
            7. Suggestions for targeting specific grant amount ranges based on the project objectives
            """
            comprehensive_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "comprehensive analysis", {"Selected Clusters": selected_clusters}, custom_context,
                comprehensive_prompt
            )
            st.markdown(comprehensive_analysis)

        st.subheader("Explore Further")
        user_question = st.text_input("Ask a question about the grant amount distribution:")
        if user_question and ai_enabled:
            user_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "user question", {"Selected Clusters": selected_clusters}, custom_context,
                user_question
            )
            st.markdown(user_analysis)

        if st.checkbox("Show Underlying Data", key="show_underlying_data02"):
            st.write(grouped_df[grouped_df['Amount Usd Cluster'].isin(selected_clusters)])

        if st.button("Download Data as CSV", key="download_data_button"):
            filtered_df = grouped_df[grouped_df['Amount Usd Cluster'].isin(selected_clusters)]
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="grants_data_distribution.csv",
                mime="text/csv"
            )

    else:
        st.warning("Please select at least one USD cluster to display the charts.")

    st.markdown("""
    This AI-powered analysis was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using advanced AI language models and data visualization techniques. 
    It leverages the Candid API, Streamlit, Plotly, and other open-source libraries to provide dynamic, 
    context-aware insights.
    """)