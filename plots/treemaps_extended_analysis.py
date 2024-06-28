import plotly.express as px
import streamlit as st
import pandas as pd
from typing import Dict, Any

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data


@handle_error
def create_treemap(grouped_data: pd.DataFrame, analyze_column: str, selected_label: str) -> px.treemap:
    """Create a treemap visualization of grant amounts."""
    fig = px.treemap(
        grouped_data,
        path=[analyze_column],
        values='Amount Usd',
        title=f"Treemap: Sum of Amount in USD by {clean_label(analyze_column)} for {selected_label} USD range",
        hover_data={'Amount Usd': ':.2f'}
    )
    fig.update_traces(
        textinfo="label+value+percent parent",
        hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percent of parent: %{percentParent:.1%}<extra></extra>'
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig


@handle_error
def filter_data(df: pd.DataFrame, selected_label: str, analyze_column: str) -> pd.DataFrame:
    """Filter and group the data based on selected USD range and analysis column."""
    if selected_label == 'All':
        filtered_data = df
    else:
        filtered_data = df[df['Amount Usd Cluster'] == selected_label]

    grouped_data = filtered_data.groupby(analyze_column)['Amount Usd'].sum().reset_index().sort_values(
        by='Amount Usd', ascending=True)

    return grouped_data


@handle_error
def get_block_insights(block_grants: pd.DataFrame) -> dict:
    """Generate insights for a selected block in the treemap."""
    insights = {
        "Total Grants": len(block_grants),
        "Total USD": format_currency(block_grants['Amount Usd'].sum()),
        "Average Grant USD": format_currency(block_grants['Amount Usd'].mean()),
        "Median Grant USD": format_currency(block_grants['Amount Usd'].median()),
        "Top Funders": block_grants.groupby('Funder Name')['Amount Usd'].sum().nlargest(5).reset_index(),
        "Top Recipients": block_grants.groupby('Recip Name')['Amount Usd'].sum().nlargest(5).reset_index(),
        "Grant Years Range": f"{block_grants['Year Issued'].min()} - {block_grants['Year Issued'].max()}",
        "Top Subjects": block_grants['Grant Subject Tran'].value_counts().nlargest(5).to_dict()
    }
    insights["Top Funders"]['Amount Usd'] = insights["Top Funders"]['Amount Usd'].apply(format_currency)
    insights["Top Recipients"]['Amount Usd'] = insights["Top Recipients"]['Amount Usd'].apply(format_currency)
    return insights


@handle_error
def display_block_insights(block_grants: pd.DataFrame, selected_block: str):
    """Display insights for a selected block in the treemap."""
    st.subheader(f"Insights for {selected_block}")

    insights = get_block_insights(block_grants)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Grants", insights["Total Grants"])
    with col2:
        st.metric("Total USD", insights["Total USD"])
    with col3:
        st.metric("Average Grant USD", insights["Average Grant USD"])
    with col4:
        st.metric("Median Grant USD", insights["Median Grant USD"])

    st.write(f"Grants in this category span from {insights['Grant Years Range']}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Funders")
        st.table(insights["Top Funders"])
    with col2:
        st.subheader("Top Recipient Organizations")
        st.table(insights["Top Recipients"])

    st.subheader("Top Subjects")
    st.write(insights["Top Subjects"])

    st.subheader("Sample Grant Descriptions")
    sample_grants = block_grants.sort_values(by='Amount Usd', ascending=False).head(5)
    for _, grant in sample_grants.iterrows():
        st.write(f"**Amount:** {format_currency(grant['Amount Usd'])}")
        st.write(f"**Description:** {grant['Grant Description']}")
        st.write("---")


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for treemaps extended analysis"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the treemap visualization. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Treemaps Extended Analysis", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def treemaps_extended_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                               ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Enhanced Treemaps by Subject, Population and Strategy")
    st.write("""
    Welcome to the AI-Enhanced Treemaps with Extended Analysis page! This interactive visualization allows you to explore 
    the distribution of grant amounts across different subjects, populations, and strategies using dynamic treemaps.
    Our AI assistant will provide insights to help you understand the hierarchical relationships and funding patterns 
    in the data, considering your project theme and goals. Select a category from the dropdown menu below the treemap 
    to view detailed insights about that specific category.
    """)

    usd_range_options = ['All'] + sorted(grouped_df['Amount Usd Cluster'].unique())

    col1, col2 = st.columns(2)

    with col1:
        analyze_column = st.radio("Select Variable for Treemap",
                                  options=['Grant Strategy Tran', 'Grant Subject Tran', 'Grant Population Tran'])

    with col2:
        selected_label = st.selectbox("Select USD Range", options=usd_range_options)

    grouped_data = filter_data(grouped_df, selected_label, analyze_column)
    fig = create_treemap(grouped_data, analyze_column, selected_label)

    st.plotly_chart(fig, use_container_width=True)

    if ai_enabled:
        treemap_analysis = generate_ai_analysis(
            grouped_df, grouped_df, selected_chart, selected_role, api_key,
            "treemap visualization",
            {"analyze_column": analyze_column, "selected_label": selected_label},
            custom_context
        )
        st.markdown(treemap_analysis)

    # Create a dropdown for selecting a specific block
    block_options = grouped_data[analyze_column].tolist()
    selected_block = st.selectbox("Select a category for detailed insights:", options=[""] + block_options)

    if selected_block:
        block_grants = grouped_df[grouped_df[analyze_column] == selected_block]

        if not block_grants.empty:
            display_block_insights(block_grants, selected_block)

            if ai_enabled:
                block_analysis = generate_ai_analysis(
                    block_grants, grouped_df, selected_chart, selected_role, api_key,
                    "selected block analysis",
                    {"analyze_column": analyze_column, "selected_block": selected_block},
                    custom_context
                )
                st.markdown(block_analysis)

            if st.button(f"Download Data for {selected_block} Category", key="download_category_data"):
                csv_data = block_grants.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"grants_data_{selected_block}.csv",
                    mime="text/csv"
                )

    if st.button(f"Download Data for {selected_label} USD Range", key="download_range_data"):
        csv_data = grouped_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"grants_data_{selected_label}.csv",
            mime="text/csv"
        )

    if ai_enabled:
        st.subheader("AI-Generated Comprehensive Analysis")
        comprehensive_prompt = f"""
        Provide a comprehensive analysis of the grant distribution visualized in the treemap for {analyze_column} 
        and the {selected_label} USD range. Include the following:
        1. Overall patterns and trends visible in the treemap
        2. Insights about the relative sizes of different categories and their significance
        3. Analysis of any notable hierarchical relationships or nested patterns
        4. Comparison of grant amounts and distributions across different {analyze_column} categories
        5. Potential implications for grant seekers and funders based on the observed patterns
        6. Recommendations for further investigation or analysis of specific areas in the treemap
        7. How the observed patterns and distributions relate to the project theme and goals
        8. Suggestions for targeting specific categories or subcategories based on the project objectives
        """
        comprehensive_analysis = generate_ai_analysis(
            grouped_df, grouped_df, selected_chart, selected_role, api_key,
            "comprehensive treemap analysis",
            {"analyze_column": analyze_column, "selected_label": selected_label},
            custom_context,
            comprehensive_prompt
        )
        st.markdown(comprehensive_analysis)

    st.subheader("Explore Further")
    user_question = st.text_input("Ask a question about the treemap visualization or specific categories:")
    if user_question and ai_enabled:
        user_analysis = generate_ai_analysis(
            grouped_df, grouped_df, selected_chart, selected_role, api_key,
            "user question",
            {"analyze_column": analyze_column, "selected_label": selected_label},
            custom_context,
            user_question
        )
        st.markdown(user_analysis)

    st.markdown("""
    This app was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using the latest methods 
    for enabling AI to Chat with Data. It also uses the Candid API, Streamlit, Plotly, and other open-source libraries. 
    Generative AI solutions such as OpenAI GPT-4 and Claude Opus were used to generate portions of the source code.
    """)
    