import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from utils.utils import clean_label, handle_error, generate_page_prompt, generate_dynamic_context, update_year_axis
from loaders.llama_index_setup import query_data


@handle_error
def create_scatter_plot(filtered_df: pd.DataFrame, marker_size: int, opacity: float, color_by: str) -> go.Figure:
    """Create an interactive scatter plot of grant amounts over time."""
    fig = px.scatter(
        filtered_df,
        x='Year Issued',
        y='Amount Usd',
        color=color_by,
        size='Amount Usd',
        hover_data=['Grant Key', 'Grant Description', 'Amount Usd', 'Funder Name', 'Recip Name'],
        opacity=opacity,
        size_max=marker_size,
        title='Grant Amount by Year'
    )

    fig.update_layout(
        xaxis_title='Year Issued',
        yaxis_title='Amount (USD)',
        legend_title_text=clean_label(color_by),
        height=600,
        updatemenus=[
            dict(
                buttons=list([
                    dict(args=[{"yaxis.type": "linear"}], label="Linear Scale", method="relayout"),
                    dict(args=[{"yaxis.type": "log"}], label="Log Scale", method="relayout")
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    return fig


@handle_error
def create_trend_line(filtered_df: pd.DataFrame) -> go.Figure:
    """Create a trend line chart showing average grant amount over time."""
    yearly_avg = filtered_df.groupby('Year Issued')['Amount Usd'].mean().reset_index()

    fig = px.line(
        yearly_avg,
        x='Year Issued',
        y='Amount Usd',
        title='Average Grant Amount Trend Over Time'
    )

    fig.update_layout(
        xaxis_title='Year Issued',
        yaxis_title='Average Grant Amount (USD)',
        height=400
    )

    return fig

@handle_error
def create_cluster_density_chart(filtered_df: pd.DataFrame) -> go.Figure:
    """Create a line chart showing the density of grants in each cluster over time."""
    cluster_counts = filtered_df.groupby(['Year Issued', 'Amount Usd Cluster']).size().reset_index(name='Count')
    
    fig = px.line(
        cluster_counts,
        x='Year Issued',
        y='Count',
        color='Amount Usd Cluster',
        title='Grant Density by Amount Cluster Over Time'
    )

    fig.update_layout(
        xaxis_title='Year Issued',
        yaxis_title='Number of Grants',
        legend_title='Amount Cluster',
        height=400
    )

    return fig


@handle_error
def create_categories_over_time(df: pd.DataFrame, category_column: str) -> go.Figure:
    """Create a stacked area chart showing the distribution of categories over time."""
    yearly_data = df.groupby(['Year Issued', category_column])['Amount Usd'].sum().reset_index()
    fig = px.area(yearly_data, x='Year Issued', y='Amount Usd', color=category_column,
                  title=f'Distribution of {category_column} Over Time')
    fig.update_layout(xaxis_title='Year', yaxis_title='Total Grant Amount (USD)')
    return fig


@handle_error
def filter_data(df: pd.DataFrame, start_year: int, end_year: int, min_amount: float, max_amount: float) -> pd.DataFrame:
    """Filter the DataFrame based on selected years and amount range."""
    filtered_df = df[
        (df['Year Issued'].astype(int) >= start_year) &
        (df['Year Issued'].astype(int) <= end_year) &
        (df['Amount Usd'] >= min_amount) &
        (df['Amount Usd'] <= max_amount)
        ]
    filtered_df = filtered_df[filtered_df['Amount Usd'].notna()]
    return filtered_df


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for grant amounts over time"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for grant amounts over time. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Grant Amount Scatter Plot", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def grant_amount_scatter_plot(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                              ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Enhanced Grant Amount and Trends Over Time")
    st.write("""
    Welcome to the AI-Enhanced Grant Amount and Trends Over Time analysis page! This interactive visualization allows you to explore 
    the distribution of grant amounts over time and analyze various funding trends. Our AI assistant will provide insights and analysis
    based on the visualizations and data presented, considering your project theme and goals.
    """)

    unique_years = sorted(df['Year Issued'].unique())
    start_year = st.number_input("Start Year", min_value=int(min(unique_years)),
                                 max_value=int(max(unique_years)), value=int(min(unique_years)))
    end_year = st.number_input("End Year", min_value=int(min(unique_years)), max_value=int(max(unique_years)),
                               value=int(max(unique_years)))

    col1, col2 = st.columns(2)
    with col1:
        min_amount = st.number_input("Minimum Grant Amount", min_value=0.0, value=0.0, step=1000.0)
    with col2:
        max_amount = st.number_input("Maximum Grant Amount", min_value=0.0, value=float(df['Amount Usd'].max()),
                                     step=1000.0)

    marker_size = st.slider("Marker Size", min_value=5, max_value=30, value=15)
    opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

    color_options = ['Amount Usd Cluster', 'Funder Type', 'Grant Subject Tran', 'Grant Population Tran',
                     'Grant Strategy Tran']
    color_by = st.selectbox("Color by", options=color_options)

    filtered_df = filter_data(grouped_df, start_year, end_year, min_amount, max_amount)

    if not filtered_df.empty:
        st.subheader("Interactive Scatter Plot")
        fig = create_scatter_plot(filtered_df, marker_size, opacity, color_by)
        st.plotly_chart(fig, use_container_width=True)

        if ai_enabled:
            scatter_analysis = generate_ai_analysis(
                filtered_df, grouped_df, selected_chart, selected_role, api_key,
                "scatter plot",
                {"start_year": start_year, "end_year": end_year, "min_amount": min_amount, "max_amount": max_amount,
                 "color_by": color_by},
                custom_context
            )
            st.markdown(scatter_analysis)

        st.subheader("Grant Density by Amount Cluster Over Time")
        density_fig = create_cluster_density_chart(filtered_df)
        st.plotly_chart(density_fig, use_container_width=True)

        if ai_enabled:
            density_analysis = generate_ai_analysis(
                filtered_df, grouped_df, selected_chart, selected_role, api_key,
                "cluster density trend",
                {"start_year": start_year, "end_year": end_year, "min_amount": min_amount, "max_amount": max_amount},
                custom_context
            )
            st.markdown(density_analysis)

        st.subheader(f"Distribution of {color_by} Over Time")
        st.write(f"""
        This chart shows how the distribution of grants across different {color_by} categories has changed over time. 
        It helps identify shifts in funding priorities or emerging trends in grant-making.
        """)
        categories_fig = create_categories_over_time(filtered_df, color_by)
        st.plotly_chart(categories_fig, use_container_width=True)

        if ai_enabled:
            categories_analysis = generate_ai_analysis(
                filtered_df, grouped_df, selected_chart, selected_role, api_key,
                "categories over time",
                {"start_year": start_year, "end_year": end_year, "min_amount": min_amount, "max_amount": max_amount,
                 "color_by": color_by},
                custom_context
            )
            st.markdown(categories_analysis)

        if ai_enabled:
            st.subheader("AI-Generated Comprehensive Analysis")
            comprehensive_prompt = f"""
            Provide a comprehensive analysis of the grant amount trends and distributions over time for the selected period 
            ({start_year} to {end_year}) and amount range (${min_amount:,.0f} to ${max_amount:,.0f}). Include the following:
            1. Overall trends in grant amounts and counts over the selected time period
            2. Notable changes or patterns in the distribution of grants across different {color_by} categories
            3. Potential factors influencing the observed trends and distributions
            4. Implications for grant seekers and funders
            5. Recommendations for further investigation or analysis
            6. How these trends and distributions relate to the project theme and goals
            7. Suggestions for timing and targeting of grant applications based on the observed patterns and project objectives
            """
            comprehensive_analysis = generate_ai_analysis(
                filtered_df, grouped_df, selected_chart, selected_role, api_key,
                "comprehensive analysis",
                {"start_year": start_year, "end_year": end_year, "min_amount": min_amount, "max_amount": max_amount,
                 "color_by": color_by},
                custom_context,
                comprehensive_prompt
            )
            st.markdown(comprehensive_analysis)

        st.subheader("Explore Further")
        user_question = st.text_input("Ask a question about the grant amount trends and distributions:")
        if user_question and ai_enabled:
            user_analysis = generate_ai_analysis(
                filtered_df, grouped_df, selected_chart, selected_role, api_key,
                "user question",
                {"start_year": start_year, "end_year": end_year, "min_amount": min_amount, "max_amount": max_amount,
                 "color_by": color_by},
                custom_context,
                user_question
            )
            st.markdown(user_analysis)

        if st.checkbox("Show Underlying Data", key="show_underlying_data03"):
            st.dataframe(filtered_df)

        if st.button("Download Data as CSV", key="download_data_button"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="grants_scatter_data.csv",
                mime="text/csv"
            )
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")

    st.markdown("""
    This AI-powered analysis was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using advanced AI language models and data visualization techniques. 
    It leverages the Candid API, Streamlit, Plotly, and other open-source libraries to provide dynamic, 
    context-aware insights.
    """)
