import streamlit as st
import os
import openai
from typing import Optional, Any

from plots.introduction import introduction
from utils.utils import handle_error, clean_label
from loaders.data_loader import load_data, preprocess_data
from plots.data_summary import ai_powered_data_summary
from plots.grant_amount_distribution import grant_amount_distribution
from plots.grant_amount_scatter_plot import grant_amount_scatter_plot
from plots.grant_amount_heatmap import grant_amount_heatmap
from plots.grant_description_word_clouds import grant_description_word_clouds
from plots.treemaps_extended_analysis import treemaps_extended_analysis
from plots.general_analysis_relationships import general_analysis_relationships
from plots.top_categories_unique_grants import top_categories_unique_grants
from automated_analysis.generate_full_report import generate_full_report


@handle_error
def clear_cache():
    st.cache_data.clear()


@handle_error
def init_session_state():
    if 'cache_initialized' not in st.session_state:
        st.session_state.cache_initialized = False

    if not st.session_state.cache_initialized:
        clear_cache()
        st.session_state.cache_initialized = True


@handle_error
def setup_openai() -> bool:
    """
    Set up OpenAI API key and return whether AI features are enabled.

    Returns:
        bool: True if AI features are enabled, False otherwise.
    """
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        openai.api_key = api_key
        st.session_state.api_key = api_key
        return True
    else:
        st.sidebar.warning(
            "AI features are disabled. Please enter your OpenAI API key to enable AI-assisted analysis.")
        return False


@handle_error
def load_and_preprocess_data(file_path: Optional[str] = None, uploaded_file: Optional[Any] = None) -> tuple:
    """
    Load and preprocess the data.

    Args:
        file_path (Optional[str]): Path to the JSON file.
        uploaded_file (Optional[Any]): Uploaded file object.

    Returns:
        tuple: Tuple containing the processed DataFrame and the grouped DataFrame.
    """
    grants = load_data(file_path=file_path, uploaded_file=uploaded_file)
    return preprocess_data(grants)


@handle_error
def main():
    init_session_state()
    st.set_page_config(page_title="GrantScope", page_icon=":chart_with_upwards_trend:")

    st.sidebar.markdown("Questions or feedback?")
    st.sidebar.markdown('<a href="mailto:dltzshz8@anonaddy.me">Contact the Developer!</a>', unsafe_allow_html=True)

    file_path = 'data/sample.json'
    uploaded_file = st.sidebar.file_uploader("Upload Candid API JSON File 10MB or less", accept_multiple_files=False,
                                             type="json")

    ai_enabled = setup_openai()
    api_key = st.session_state.get('api_key') if ai_enabled else None

    df, grouped_df = load_and_preprocess_data(file_path=file_path, uploaded_file=uploaded_file)

    if 'page' not in st.session_state:
        st.session_state.page = "introduction"

    user_roles = ["Grant Analyst/Writer", "Normal Grant User"]
    selected_role = st.sidebar.selectbox("Select User Role", options=user_roles)

    # Add new button for Automated Analysis
    if st.sidebar.button("Generate Full Report"):
        st.session_state.page = "automated_analysis"

    if st.session_state.page == "introduction":
        introduction(df, grouped_df, ai_enabled, api_key)
    elif st.session_state.page == "automated_analysis":
        generate_full_report()
    else:
        chart_options = {
            "Grant Analyst/Writer": [
                "Data Summary",
                "Grant Amount Distribution",
                "Grant Amount Scatter Plot",
                "Grant Amount Heatmap",
                "Grant Description Word Clouds",
                "Treemaps with Extended Analysis",
                "General Analysis of Relationships",
                "Top Categories by Unique Grant Count"
            ],
            "Normal Grant User": [
                "Data Summary",
                "Grant Amount Distribution",
                "Grant Amount Scatter Plot",
                "Grant Amount Heatmap",
                "Grant Description Word Clouds",
                "Treemaps with Extended Analysis"
            ]
        }

        selected_chart = st.sidebar.selectbox("Select Chart", options=chart_options[selected_role])

        st.title("GrantScope Dashboard")

        # Call the appropriate function based on the selected chart
        if selected_chart == "Data Summary":
            ai_powered_data_summary(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Grant Amount Distribution":
            grant_amount_distribution(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Grant Amount Scatter Plot":
            grant_amount_scatter_plot(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Grant Amount Heatmap":
            grant_amount_heatmap(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Grant Description Word Clouds":
            grant_description_word_clouds(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Treemaps with Extended Analysis":
            treemaps_extended_analysis(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "General Analysis of Relationships":
            general_analysis_relationships(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)
        elif selected_chart == "Top Categories by Unique Grant Count":
            top_categories_unique_grants(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key)


if __name__ == '__main__':
    main()
