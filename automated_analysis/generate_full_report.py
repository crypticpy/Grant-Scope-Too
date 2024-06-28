import streamlit as st
from loaders.data_loader import load_data, preprocess_data
from loaders.llama_index_setup import query_data, setup_llama_index
from plots.data_summary import ai_powered_data_summary
from plots.grant_amount_distribution import grant_amount_distribution
from plots.grant_amount_scatter_plot import grant_amount_scatter_plot
from plots.grant_amount_heatmap import grant_amount_heatmap
from plots.grant_description_word_clouds import grant_description_word_clouds
from plots.treemaps_extended_analysis import treemaps_extended_analysis
from plots.general_analysis_relationships import general_analysis_relationships
from plots.top_categories_unique_grants import top_categories_unique_grants
from utils.utils import handle_error, generate_dynamic_context
import inspect


def run_analysis_section(analysis_function, df, grouped_df, selected_chart, selected_role, ai_enabled, api_key, project_theme):
    # Create a custom context that includes the project theme
    custom_context = {
        "project_theme": project_theme,
        "analysis_type": analysis_function.__name__.replace('_', ' ').title()
    }
    
    # Run the analysis function with the custom context
    return analysis_function(df, grouped_df, selected_chart, selected_role, ai_enabled, api_key, custom_context)

@handle_error
def generate_full_report():
    st.title("Comprehensive Grant Analysis Report")

    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.error("API key is not set. Please enter your API key in the sidebar.")
        return

    project_theme = st.text_area("Enter your project theme and objectives:", key="project_theme_input")
    uploaded_file = st.file_uploader("Upload your JSON data file", type="json", key="json_file_uploader")

    if uploaded_file is None:
        st.warning("Please upload a JSON data file to proceed.")
        return

    if st.button("Generate Full Report", key="generate_report_button"):
        with st.spinner("Loading and preprocessing data..."):
            grants = load_data(uploaded_file=uploaded_file)
            df, grouped_df = preprocess_data(grants)

        analysis_functions = [
            ai_powered_data_summary,
            grant_amount_distribution,
            grant_amount_scatter_plot,
            grant_amount_heatmap,
            grant_description_word_clouds,
            treemaps_extended_analysis,
            general_analysis_relationships,
            top_categories_unique_grants
        ]

        all_analyses = []
        progress_bar = st.progress(0)

        for i, func in enumerate(analysis_functions):
            progress_bar.progress((i + 1) / len(analysis_functions))
            st.subheader(f"{i + 1}. {func.__name__.replace('_', ' ').title()}")

            with st.spinner(f"Generating analysis for {func.__name__}..."):
                analysis_result = run_analysis_section(
                    func, df, grouped_df, "Automated Report", "Grant Analyst", True, st.session_state.api_key, project_theme
                )
                all_analyses.append(str(analysis_result))

        st.subheader("Final Summary and Recommendations")
        with st.spinner("Generating final summary and recommendations..."):
            final_summary = generate_final_summary(df, project_theme, all_analyses, st.session_state.api_key)
            st.markdown(final_summary)

        st.success("Comprehensive analysis complete!")


@st.cache_data
def generate_final_summary(df, project_theme, all_analyses, api_key):
    setup_llama_index(api_key)

    combined_analysis = "\n\n".join([analysis for analysis in all_analyses if analysis])

    final_prompt = f"""
    Project Theme: {project_theme}

    Based on the comprehensive analysis of the grant data and the project theme, provide a final summary and recommendations:

    {combined_analysis}

    Please address the following in your summary:
    1. How does the analyzed data relate to the project theme?
    2. What are the most relevant insights for this specific project?
    3. What funding opportunities or strategies would you recommend based on the data and the project theme?
    4. Are there any potential challenges or areas of concern given the project theme and the grant landscape?
    5. Provide actionable recommendations for the grant seeker based on this analysis.

    Organize your response into clear sections with headers for easy reading. Make sure to use the available data to support your recommendations. 
    """

    query_result = query_data(df, final_prompt, "Final Summary and Recommendations", {}, api_key)

    if isinstance(query_result, dict):
        final_summary = query_result.get('response', str(query_result))
    elif isinstance(query_result, str):
        final_summary = query_result
    else:
        final_summary = str(query_result)

    return final_summary


if __name__ == "__main__":
    generate_full_report()