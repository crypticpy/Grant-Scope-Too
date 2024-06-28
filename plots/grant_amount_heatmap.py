import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Any

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data


@handle_error
def create_heatmap(pivot_table: pd.DataFrame, dimension1: str, dimension2: str) -> go.Figure:
    """
    Create a heatmap of grant amounts based on two dimensions.
    """
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Total Grant Amount: $%{z:,.0f}<extra></extra>',
        colorbar=dict(title='Total Grant Amount')
    ))

    fig.update_layout(
        title=f'Total Grant Amount by {clean_label(dimension1)} and {clean_label(dimension2)}',
        xaxis_title=clean_label(dimension2),
        yaxis_title=clean_label(dimension1),
        width=900,
        height=700
    )

    return fig


@handle_error
def filter_and_pivot_data(df: pd.DataFrame, dimension1: str, dimension2: str, selected_values1: List[str],
                          selected_values2: List[str]) -> pd.DataFrame:
    """
    Filter and pivot the data based on selected dimensions and values.
    """
    filtered_df = df[
        df[dimension1].isin(selected_values1) &
        df[dimension2].isin(selected_values2)
        ]

    pivot_table = filtered_df.groupby([dimension1, dimension2])['Amount Usd'].sum().unstack().fillna(0)
    return pivot_table


@handle_error
def create_top_combinations_chart(df: pd.DataFrame, dimension1: str, dimension2: str, top_n: int = 10) -> go.Figure:
    """
    Create a bar chart of top combinations by total grant amount.
    """
    combinations = df.groupby([dimension1, dimension2])['Amount Usd'].sum().sort_values(ascending=False).head(top_n)

    fig = go.Figure(data=[
        go.Bar(
            x=[f"{d1} - {d2}" for d1, d2 in combinations.index],
            y=combinations.values,
            text=[f"${x:,.0f}" for x in combinations.values],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=f'Top {top_n} {clean_label(dimension1)} - {clean_label(dimension2)} Combinations by Total Grant Amount',
        xaxis_title='Combination',
        yaxis_title='Total Grant Amount (USD)',
        width=900,
        height=500
    )

    return fig


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for grant amount heatmap"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the grant amount heatmap. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Grant Amount Heatmap", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def grant_amount_heatmap(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Enhanced Grant Amount Heatmap Analysis")
    st.write("""
    Welcome to the AI-Enhanced Grant Amount Heatmap Analysis page! This interactive tool allows you to explore the intersection 
    of grant dimensions and identify meaningful funding patterns. The heatmap visualizes the concentration of grant 
    amounts across different combinations of dimensions, while our AI assistant provides insights to help you uncover 
    trends about funding priorities and patterns, considering your project theme and goals.
    """)

    dimension_options = ['Grant Subject Tran', 'Grant Population Tran', 'Grant Strategy Tran', 'Funder Type']
    default_dim1, default_dim2 = dimension_options[:2]

    col1, col2 = st.columns(2)
    with col1:
        dimension1 = st.selectbox("Select Dimension 1", options=dimension_options,
                                  index=dimension_options.index(default_dim1))
    with col2:
        dimension2 = st.selectbox("Select Dimension 2", options=[d for d in dimension_options if d != dimension1],
                                  index=0)

    st.caption("Select individual values for each dimension to filter the heatmap.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"Select {clean_label(dimension1)}s", expanded=False):
            selected_values1 = st.multiselect(f"Select {clean_label(dimension1)}s",
                                              options=grouped_df[dimension1].unique(),
                                              default=grouped_df[dimension1].unique())
    with col2:
        with st.expander(f"Select {clean_label(dimension2)}s", expanded=False):
            selected_values2 = st.multiselect(f"Select {clean_label(dimension2)}s",
                                              options=grouped_df[dimension2].unique(),
                                              default=grouped_df[dimension2].unique())

    pivot_table = filter_and_pivot_data(grouped_df, dimension1, dimension2, selected_values1, selected_values2)

    if not pivot_table.empty:
        st.subheader("Grant Amount Heatmap")
        fig = create_heatmap(pivot_table, dimension1, dimension2)
        st.plotly_chart(fig, use_container_width=True)

        if ai_enabled:
            heatmap_analysis = generate_ai_analysis(
                grouped_df, grouped_df, selected_chart, selected_role, api_key,
                "heatmap",
                {"dimension1": dimension1, "dimension2": dimension2, "selected_values1": selected_values1,
                 "selected_values2": selected_values2},
                custom_context
            )
            st.markdown(heatmap_analysis)

        st.subheader(f"Top {clean_label(dimension1)} - {clean_label(dimension2)} Combinations")
        top_combinations_fig = create_top_combinations_chart(grouped_df, dimension1, dimension2)
        st.plotly_chart(top_combinations_fig, use_container_width=True)

        if ai_enabled:
            combinations_analysis = generate_ai_analysis(
                grouped_df, grouped_df, selected_chart, selected_role, api_key,
                "top combinations chart",
                {"dimension1": dimension1, "dimension2": dimension2},
                custom_context
            )
            st.markdown(combinations_analysis)

        if ai_enabled:
            st.subheader("AI-Generated Comprehensive Analysis")
            comprehensive_prompt = f"""
            Provide a comprehensive analysis of the grant amount distribution across the selected dimensions 
            ({clean_label(dimension1)} and {clean_label(dimension2)}). Include the following:
            1. Overall patterns and trends visible in the heatmap
            2. Notable high-concentration and low-concentration areas
            3. Insights about the relationship between {clean_label(dimension1)} and {clean_label(dimension2)}
            4. Analysis of the top combinations and their significance
            5. Potential implications for grant seekers and funders based on the observed patterns
            6. Recommendations for further investigation or analysis
            7. How these patterns and combinations relate to the project theme and goals
            8. Suggestions for targeting specific combinations or areas based on the project objectives
            """
            comprehensive_analysis = generate_ai_analysis(
                grouped_df, grouped_df, selected_chart, selected_role, api_key,
                "comprehensive analysis",
                {"dimension1": dimension1, "dimension2": dimension2, "selected_values1": selected_values1,
                 "selected_values2": selected_values2},
                custom_context,
                comprehensive_prompt
            )
            st.markdown(comprehensive_analysis)

        st.subheader("Explore Further")
        user_question = st.text_input("Ask a question about the grant amount heatmap or top combinations:")
        if user_question and ai_enabled:
            user_analysis = generate_ai_analysis(
                grouped_df, grouped_df, selected_chart, selected_role, api_key,
                "user question",
                {"dimension1": dimension1, "dimension2": dimension2, "selected_values1": selected_values1,
                 "selected_values2": selected_values2},
                custom_context,
                user_question
            )
            st.markdown(user_analysis)

        if st.checkbox("Show Underlying Data", key="show_underlying_data"):
            st.write(pivot_table)

        if st.button("Download Data as CSV", key="download_data_button"):
            csv_data = pivot_table.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="heatmap_data.csv",
                mime="text/csv"
            )

    else:
        st.warning("No data available for the selected dimensions and values. Please adjust your selection.")

    st.markdown("""
    This AI-powered analysis was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using advanced AI language models and data visualization techniques. 
    It leverages the Candid API, Streamlit, Plotly, and other open-source libraries to provide dynamic, 
    context-aware insights.
    """)