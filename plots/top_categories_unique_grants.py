import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any
from textwrap import shorten
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data


@handle_error
def get_column_name(df: pd.DataFrame, possible_names: List[str]) -> str:
    """Get the correct column name from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the column names {possible_names} found in the DataFrame.")


@handle_error
def create_chart(normalized_counts: pd.DataFrame, top_n: int, chart_type: str, categorical_col: str) -> go.Figure:
    """Create a chart based on the specified type."""
    if chart_type == "Bar Chart":
        fig = px.bar(normalized_counts.head(top_n), x='Unique Grant Keys', y='truncated_col', orientation='h',
                     title=f"Top {top_n} Categories in {clean_label(categorical_col)}",
                     hover_data={categorical_col: True})
        fig.update_layout(yaxis_title=clean_label(categorical_col), xaxis_title='Number of Unique Grants',
                          height=600, margin=dict(l=0, r=0, t=30, b=0))
    elif chart_type == "Pie Chart":
        fig = px.pie(normalized_counts.head(top_n), values='Unique Grant Keys', names='truncated_col',
                     title=f"Distribution of Unique Grant Keys Across Top {top_n} Categories in {clean_label(categorical_col)}")
        fig.update_layout(height=600)
    else:  # Treemap
        fig = px.treemap(normalized_counts.head(top_n), path=['truncated_col'], values='Unique Grant Keys',
                         title=f"Treemap of Unique Grant Keys Across Top {top_n} Categories in {clean_label(categorical_col)}")
        fig.update_layout(height=600)
    return fig


@handle_error
def display_category_details(df: pd.DataFrame, categorical_col: str, selected_category: str):
    """Display details for a selected category."""
    category_grants = df[df[categorical_col] == selected_category].drop_duplicates(subset=['Grant Key'])

    if not category_grants.empty:
        st.write(f"### Grant Details for {selected_category}:")
        st.write(f"Total Grants: {len(category_grants)}")
        st.write(f"Total Amount: {format_currency(category_grants['Amount Usd'].sum())}")
        st.write(f"Average Amount: {format_currency(category_grants['Amount Usd'].mean())}")

        st.subheader("Top Funders")
        top_funders = category_grants.groupby('Funder Name')['Amount Usd'].sum().nlargest(5).reset_index()
        top_funders['Amount Usd'] = top_funders['Amount Usd'].apply(format_currency)
        st.table(top_funders)

        st.subheader("Sample Grants")
        sample_grants = category_grants.sample(min(5, len(category_grants)))
        for _, grant in sample_grants.iterrows():
            st.write(f"**Amount:** {format_currency(grant['Amount Usd'])}")
            st.write(f"**Funder:** {grant['Funder Name']}")
            st.write(f"**Recipient:** {grant['Recip Name']}")
            st.write(f"**Description:** {grant['Grant Description']}")
            st.write("---")
    else:
        st.write(f"No grants found for the selected category: {selected_category}")


@handle_error
def create_excel_file(df: pd.DataFrame, normalized_counts: pd.DataFrame, categorical_col: str,
                      top_n: int) -> io.BytesIO:
    """Create an Excel file with multiple sheets: one for the summary data and one for each top category."""
    output = io.BytesIO()
    workbook = Workbook()

    # Summary sheet
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    for r in dataframe_to_rows(normalized_counts.head(top_n), index=False, header=True):
        summary_sheet.append(r)

    # Individual category sheets
    top_categories = normalized_counts.head(top_n)[categorical_col].tolist()
    for category in top_categories:
        sheet = workbook.create_sheet(title=str(category)[:31])  # Excel sheet names are limited to 31 characters
        category_data = df[df[categorical_col] == category].sort_values('Amount Usd', ascending=False)
        for r in dataframe_to_rows(category_data, index=False, header=True):
            sheet.append(r)

    workbook.save(output)
    output.seek(0)
    return output


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for top categories by unique grant count"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the top categories by unique grant count. Provide insights, trends, and implications from the perspective of an experienced grant consultant to a client seeking funding advice based on this data"

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Top Categories by Unique Grant Count", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def top_categories_unique_grants(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                                 ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Enhanced Top Categories by Unique Grant Count")
    st.write("""
    Welcome to the AI-Enhanced Top Categories by Unique Grant Count page! This interactive visualization allows you to explore
    the distribution of unique grant counts across different categorical variables. Our AI assistant will provide insights 
    to help you understand patterns and trends in the data, considering your project theme and goals. You can analyze categories 
    such as funder type, recipient organization, grant subject, population served, funding strategy, or year of issuance.
    """)

    key_categorical_columns = [
        ['Funder Type', 'Funder Type'],
        ['Recip Organization Tran', 'Recip Organization Tran'],
        ['Grant Subject Tran', 'Grant Subject Tran'],
        ['Grant Population Tran', 'Grant Population Tran'],
        ['Grant Strategy Tran', 'Grant Strategy Tran'],
        ['Year Issued', 'Year Issued']
    ]

    col1, col2 = st.columns(2)

    with col1:
        selected_categorical = st.selectbox("Select Categorical Variable",
                                            options=[col[0] for col in key_categorical_columns])
        top_n = st.slider("Number of Top Categories", min_value=5, max_value=20, value=10, step=1)

    with col2:
        chart_type = st.selectbox("Select Chart Type", options=["Bar Chart", "Pie Chart", "Treemap"])
        sort_order = st.radio("Sort Order", options=["Descending", "Ascending"])

    try:
        categorical_col = get_column_name(df,
                                          [col[1] for col in key_categorical_columns if col[0] == selected_categorical])
        normalized_counts = df.groupby(categorical_col)['Grant Key'].nunique().sort_values(
            ascending=(sort_order == "Ascending")).reset_index()

        normalized_counts.columns = [categorical_col, 'Unique Grant Keys']

        normalized_counts['truncated_col'] = normalized_counts[categorical_col].apply(
            lambda x: shorten(str(x), width=30, placeholder="..."))

        fig = create_chart(normalized_counts, top_n, chart_type, categorical_col)
        st.plotly_chart(fig, use_container_width=True)

        if ai_enabled:
            chart_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                f"{chart_type} of top categories",
                {"categorical_col": categorical_col, "top_n": top_n, "chart_type": chart_type, "sort_order": sort_order},
                custom_context
            )
            st.markdown(chart_analysis)

        st.write(
            f"Top {top_n} Categories account for {normalized_counts.head(top_n)['Unique Grant Keys'].sum() / normalized_counts['Unique Grant Keys'].sum():.2%} of total unique grants")

        st.subheader("Category Details")
        selected_category = st.selectbox(f"Select {selected_categorical} Category",
                                         options=normalized_counts[categorical_col])

        display_category_details(df, categorical_col, selected_category)

        if ai_enabled:
            category_analysis = generate_ai_analysis(
                df[df[categorical_col] == selected_category], grouped_df, selected_chart, selected_role, api_key,
                "selected category analysis",
                {"categorical_col": categorical_col, "selected_category": selected_category},
                custom_context
            )
            st.markdown(category_analysis)

        if st.button("Download Detailed Data for Top Categories", key="download_detailed_data"):
            excel_file = create_excel_file(df, normalized_counts, categorical_col, top_n)
            st.download_button(
                label="Download Excel File",
                data=excel_file,
                file_name=f"top_{top_n}_categories_{categorical_col}_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        if ai_enabled:
            st.subheader("AI-Generated Comprehensive Analysis")
            comprehensive_prompt = f"""
            As a very experienced grant consultant, provide a comprehensive analysis of the {top_n} categories for {selected_categorical}. Include the following:
            1. Overall patterns and trends in the distribution of unique grants across categories
            2. Insights you see in the data about the relative sizes of different categories and their significance
            3. Analysis of any notable differences between the top categories and lower-ranked ones and why that might be the case
            4. Potential factors influencing the observed distribution of unique grants, considering the funders, and other data provided here, but also including known world factors and events
            5. Implications for grant seekers based on the category distribution and details observed or that can be logically inferred
            6. Recommendations for further investigation or analysis of specific categories or trends by the funding seeker
            7. How these trends and patterns relate to the project theme and goals
            8. Suggestions for targeting specific categories or strategies based on the project objectives and the observed distribution
            """
            comprehensive_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "comprehensive category analysis",
                {"categorical_col": categorical_col, "top_n": top_n, "chart_type": chart_type,
                 "sort_order": sort_order},
                custom_context,
                comprehensive_prompt
            )
            st.markdown(comprehensive_analysis)

        st.subheader("Explore Further")
        user_question = st.text_input("Ask a question about the top categories or unique grant distribution:")
        if user_question and ai_enabled:
            user_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "user question",
                {"categorical_col": categorical_col, "top_n": top_n, "chart_type": chart_type,
                 "sort_order": sort_order},
                custom_context,
                user_question
            )
            st.markdown(user_analysis)

    except ValueError as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try selecting a different categorical variable.")

    st.markdown("""
    This AI-powered analysis was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using advanced AI language models and data visualization techniques. 
    It leverages the Candid API, Streamlit, Plotly, and other open-source libraries to provide dynamic, 
    context-aware insights.
    """)
    