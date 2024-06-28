import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from typing import List, Set, Dict, Any
import plotly.express as px
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils.utils import clean_label, format_currency, handle_error, generate_page_prompt, generate_dynamic_context
from loaders.llama_index_setup import query_data

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


@st.cache_data
def load_and_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Load and process the data for word cloud analysis."""
    df['Description Word Count'] = df['Grant Description'].apply(lambda x: len(str(x).split()))
    return df


@handle_error
def create_word_cloud(text: str, stopwords: Set[str], title: str) -> plt.Figure:
    """Create a word cloud from the given text."""
    wordcloud = WordCloud(stopwords=stopwords, width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)

    return fig


@handle_error
def get_top_values(df: pd.DataFrame, column: str, n: int = 5) -> List[str]:
    """Get the top n most frequent values in a DataFrame column."""
    return df[column].value_counts().nlargest(n).index.tolist()


@handle_error
def get_top_words(text: str, n: int = 10) -> Dict[str, int]:
    """Get the top n most frequent words in the text, excluding stopwords."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    word_freq = Counter(word for word in words if word.isalnum() and word not in stop_words)
    return dict(word_freq.most_common(n))


@handle_error
def create_word_frequency_chart(word_freq: Dict[str, int]) -> px.bar:
    """Create a bar chart of word frequencies."""
    df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    fig = px.bar(df, x='Word', y='Frequency', title='Top Words Frequency')
    fig.update_layout(xaxis_title='Word', yaxis_title='Frequency')
    return fig


@handle_error
def generate_ai_analysis(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         api_key: str, analysis_type: str, context: Dict[str, Any], custom_context: Any,
                         custom_prompt: str = None) -> str:
    """Generate AI analysis based on the provided context and prompt."""
    additional_context = f"the {analysis_type} for grant description word clouds"
    pre_prompt = generate_page_prompt(df, grouped_df, selected_chart, selected_role, additional_context)

    if custom_prompt:
        query = custom_prompt
    else:
        query = f"Analyze the {analysis_type} for the grant descriptions. Provide insights, trends, and implications."

    if isinstance(custom_context, dict) and 'project_theme' in custom_context:
        query += f" Consider the project theme: {custom_context['project_theme']}"

    dynamic_context = generate_dynamic_context(df, "Grant Description Word Clouds", analysis_type,
                                               context,
                                               st.session_state.get('user_interactions', []),
                                               custom_context if isinstance(custom_context, dict) else None)

    response = query_data(df, query, pre_prompt, dynamic_context, api_key)
    return response


@handle_error
def grant_description_word_clouds(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                                  ai_enabled: bool, api_key: str, custom_context: Any = None) -> None:
    st.header("AI-Enhanced Grant Description Word Clouds and Analysis")
    st.write("""
    Welcome to the AI-Enhanced Grant Description Word Clouds and Analysis page! This interactive tool allows you to explore 
    the most common words and themes in grant descriptions. You can generate word clouds for different subsets of 
    the data, analyze word frequencies, and receive AI-powered insights about the trends and patterns in grant descriptions,
    all while considering your specific project theme and goals.
    """)

    df = load_and_process_data(df)

    stopwords = set(STOPWORDS)
    additional_stopwords = {'public', 'Public', 'health', 'Health', 'and', 'And', 'to', 'To', 'of', 'Of', 'the',
                            'The', 'a', 'A', 'by', 'By', 'in', 'In', 'for', 'For', 'with', 'With', 'on', 'On', 'is',
                            'Is', 'that', 'That', 'are', 'Are', 'as', 'As', 'be', 'Be', 'this', 'This', 'will', 'Will',
                            'at', 'Generally', 'generally', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which',
                            'At', 'from', 'From', 'or', 'Or', 'an', 'An', 'which', 'Which', 'have', 'Have', 'it',
                            'It', 'general', 'General', 'GENERAL', 'can', 'Can', 'more', 'More', 'has', 'Has', 'their',
                            'not', 'Not', 'who', 'Who', 'their', 'Their', 'we', 'We', 'support', 'Support',
                            'project', 'grant', 'GRANT', 'Grant', 'funding', 'funded', 'funds', 'fund', 'funder',
                            'recipient', 'area', 'Project'}
    stopwords.update(additional_stopwords)

    cloud_basis_options = ['Entire Dataset', 'Subject', 'Population', 'Strategy', 'Funder', 'Recipient',
                           'Geographical Area', 'Amount Usd Cluster']
    selected_basis = st.selectbox("Select the basis for generating word clouds:", options=cloud_basis_options)

    if selected_basis == 'Entire Dataset':
        text = ' '.join(df['Grant Description'])
        fig = create_word_cloud(text, stopwords, 'Word Cloud for Entire Dataset')
        st.pyplot(fig)
        plt.close(fig)

        top_words = get_top_words(text)
        st.subheader("Top Words Frequency")
        freq_fig = create_word_frequency_chart(top_words)
        st.plotly_chart(freq_fig)

        if ai_enabled:
            entire_dataset_analysis = generate_ai_analysis(
                df, grouped_df, selected_chart, selected_role, api_key,
                "entire dataset word cloud and frequency chart",
                {"basis": "Entire Dataset"},
                custom_context
            )
            st.markdown(entire_dataset_analysis)
    else:
        column_mapping = {
            'Subject': 'Grant Subject Tran',
            'Population': 'Grant Population Tran',
            'Strategy': 'Grant Strategy Tran',
            'Funder': 'Funder Name',
            'Recipient': 'Recip Name',
            'Geographical Area': 'Grant Geo Area Tran',
            'Amount Usd Cluster': 'Amount Usd Cluster'
        }
        selected_column = column_mapping[selected_basis]

        top_values = get_top_values(df, selected_column)
        for value in top_values:
            filtered_df = df[df[selected_column] == value]
            text = ' '.join(filtered_df['Grant Description'])
            fig = create_word_cloud(text, stopwords, f'Word Cloud for {selected_basis}: {value}')
            st.pyplot(fig)
            plt.close(fig)

            top_words = get_top_words(text)
            st.subheader(f"Top Words Frequency for {value}")
            freq_fig = create_word_frequency_chart(top_words)
            st.plotly_chart(freq_fig)

            if ai_enabled:
                category_analysis = generate_ai_analysis(
                    filtered_df, grouped_df, selected_chart, selected_role, api_key,
                    f"{selected_basis} category word cloud and frequency chart",
                    {"basis": selected_basis, "category": value},
                    custom_context
                )
                st.markdown(category_analysis)

    if ai_enabled:
        st.subheader("AI-Generated Comprehensive Analysis")
        comprehensive_prompt = f"""
        Provide a comprehensive analysis of the grant descriptions based on the word clouds and frequency charts. Include the following:
        1. Overall themes and trends visible in the word clouds
        2. Insights about the most frequent words and their significance
        3. Comparison of word usage across different {selected_basis} categories (if applicable)
        4. Potential implications for grant writing and funding priorities
        5. Recommendations for grant seekers based on the observed language patterns
        6. Suggestions for further text analysis or investigation
        7. How the observed themes and language patterns relate to the project theme and goals
        8. Recommendations for tailoring grant applications based on the common language and themes identified
        """
        comprehensive_analysis = generate_ai_analysis(
            df, grouped_df, selected_chart, selected_role, api_key,
            "comprehensive word cloud analysis",
            {"basis": selected_basis},
            custom_context,
            comprehensive_prompt
        )
        st.markdown(comprehensive_analysis)

    st.subheader("Explore Further")
    user_question = st.text_input("Ask a question about the grant descriptions or word usage patterns:")
    if user_question and ai_enabled:
        user_analysis = generate_ai_analysis(
            df, grouped_df, selected_chart, selected_role, api_key,
            "user question",
            {"basis": selected_basis},
            custom_context,
            user_question
        )
        st.markdown(user_analysis)

    st.subheader("Search Grant Descriptions")
    st.write("Enter words to search all grant descriptions for specific terms.")
    input_text = st.text_input("Enter word(s) to search (separate multiple words with commas):")

    if input_text:
        search_terms = [term.strip() for term in input_text.split(',')]
        grant_descriptions = df[
            df['Grant Description'].apply(lambda x: all(term.lower() in x.lower() for term in search_terms))]

        if not grant_descriptions.empty:
            st.write(f"Grant Descriptions containing all of the following terms: {', '.join(search_terms)}")
            grant_details = grant_descriptions[
                ['Grant Key', 'Grant Description', 'Funder Name', 'Funder City', 'Funder Profile Url', 'Amount Usd']]
            st.dataframe(grant_details)

            if st.button("Download Search Results as CSV", key="download_search_results"):
                csv_data = grant_details.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="search_results.csv",
                    mime="text/csv"
                )

            if ai_enabled:
                search_analysis = generate_ai_analysis(
                    grant_descriptions, grouped_df, selected_chart, selected_role, api_key,
                    "search results analysis",
                    {"search_terms": search_terms},
                    custom_context
                )
                st.markdown(search_analysis)
        else:
            st.write(f"No grant descriptions found containing all of the following terms: {', '.join(search_terms)}.")

    st.markdown("""
    This AI-powered analysis was produced by [Christopher Collins](https://www.linkedin.com/in/cctopher/) using advanced AI language models and data visualization techniques. 
    It leverages the Candid API, Streamlit, Plotly, and other open-source libraries to provide dynamic, 
    context-aware insights.
    """)