from typing import Tuple, Dict, Any

import pandas as pd
import plotly.express as px
import streamlit as st

from loaders.llama_index_setup import query_data
from utils.utils import clean_label, handle_error
from utils.generate_full_analysis_helper import generate_full_analysis

@handle_error
def generate_data_synopsis(df: pd.DataFrame, api_key: str) -> str:
    """Generate a brief AI-powered synopsis of the dataset."""
    synopsis_prompt = """
    Provide a brief, engaging synopsis of the grant dataset. Include key statistics such as total grants, 
    total amount, number of unique funders and recipients. Highlight any interesting patterns or trends. 
    Keep the language simple and charming, as if explaining to a curious 5-year-old.
    """
    context = {
        "total_grants": len(df),
        "total_amount": df['Amount Usd'].sum(),
        "unique_funders": df['Funder Name'].nunique(),
        "unique_recipients": df['Recip Name'].nunique(),
    }
    return query_data(df, synopsis_prompt, "Generate a data synopsis", context, api_key)


@handle_error
def create_simple_chart(df: pd.DataFrame, x_axis: str, y_axis: str) -> Tuple[px.bar, pd.DataFrame]:
    """Create a simple bar chart based on user selection and return the chart data."""
    if y_axis == 'Grant Key':
        chart_df = df.groupby(x_axis).size().reset_index(name='Count')
        fig = px.bar(chart_df, x=x_axis, y='Count', title=f"Number of Grants by {clean_label(x_axis)}")
    else:
        chart_df = df.groupby(x_axis)[y_axis].sum().reset_index()
        fig = px.bar(chart_df, x=x_axis, y=y_axis, title=f"Total {clean_label(y_axis)} by {clean_label(x_axis)}")

    fig.update_layout(xaxis_title=clean_label(x_axis), yaxis_title=clean_label(y_axis))
    return fig, chart_df


@handle_error
def generate_chart_insight(chart_df: pd.DataFrame, x_axis: str, y_axis: str, api_key: str) -> str:
    """Generate AI insight based on the user-created chart using LlamaIndex."""
    insight_prompt = f"""
    Analyze the chart data showing {clean_label(y_axis)} by {clean_label(x_axis)}. Please provide:
    1. The top 3 {x_axis} categories by {y_axis} and their values.
    2. The lowest {x_axis} category by {y_axis} and its value.
    3. Any noticeable patterns or trends in the data.
    4. One interesting insight or implication based on this distribution.
    Keep the explanation simple and engaging, as if talking to a curious newcomer to grant analysis.
    Start by introducing yourself, thanking the user for using the GrantScope application and then introduce the chart and analysis
    """
    return query_data(chart_df, insight_prompt, f"Chart analysis of {y_axis} by {x_axis}", {}, api_key)


@handle_error
def introduction(df: pd.DataFrame, grouped_df: pd.DataFrame, ai_enabled: bool, api_key: str) -> None:
    if 'intro_stage' not in st.session_state:
        st.session_state.intro_stage = 0

    st.title("Welcome to GrantScope: Your AI-Powered Grant Analysis Companion! 🚀🔍")

    if st.session_state.intro_stage == 0:
        st.write("""
        Hello there, curious explorer! 👋 You're about to embark on an exciting journey through the world of grants 
        with GrantScope. Imagine having a super-smart robot friend who can read through thousands of grant documents 
        in the blink of an eye and tell you all sorts of cool things about them. That's what GrantScope does!

        Here's what makes GrantScope special:
        - 🧠 AI-Powered Analysis: We use advanced artificial intelligence to understand and explain grant data.
        - 📊 Interactive Visualizations: You can play with colorful charts and graphs to see the data in fun ways.
        - 🔎 Deep Insights: Our AI can answer your questions about the data, just like a knowledgeable friend would.
        - 🛠 Cutting-Edge Tech: We use cool tools like Streamlit, Plotly, and LlamaIndex to make all this magic happen.

        Ready to dive in? First, let's load some data!
        """)

        st.info(
            "👈 Use the sidebar on the left to upload your own data then click 'Continue', or simply click 'Continue' to load sample data.")

        if st.button("Continue the Journey", key="continue_the_journey_button0"):
            st.session_state.intro_stage = 1
            st.experimental_rerun()

    elif st.session_state.intro_stage == 1:
        st.write("Great! We've got our data loaded. Now, let's see what our AI friend can tell us about it.")
        st.info("Click the button below to activate AI superpowers")
        if st.button("Generate AI Data Synopsis", key="generate_ai_data_synopsis_button0"):
            with st.spinner("Our AI is cooking up a tasty data summary for you..."):
                synopsis = generate_data_synopsis(df, api_key)
            st.markdown(synopsis)
            st.session_state.intro_stage = 2
            st.info("Click 'Continue' to build a chart with your data")
            st.button("Continue to Chart Creation", key="continue_to_chart_creation_button01")


    elif st.session_state.intro_stage == 2:
        st.subheader("Turbocharged-AI Chart Analysis!")
        st.write(
            "Explore the analysis or choose what you want to see on the x-axis and y-axis of the chart. The AI will adjust it's analysis as you interact with the chart")
        x_options = ['Funder Type', 'Grant Subject Tran', 'Grant Population Tran', 'Year Issued']
        y_options = ['Amount Usd', 'Grant Key']
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("Choose X-axis", options=x_options)
        with col2:
            y_axis = st.selectbox("Choose Y-axis", options=y_options)
        if x_axis and y_axis:
            fig, chart_df = create_simple_chart(df, x_axis, y_axis)
            st.plotly_chart(fig)

            if ai_enabled:
                with st.spinner("AI is analyzing your chart..."):
                    st.info(
                        "After the analysis loads, try changing the axes to see how the analysis changes.")
                    chart_insight = generate_chart_insight(chart_df, x_axis, y_axis, api_key)
                st.subheader("Artificial Intelligence Data Analysis: Complete")
                st.write("The information below is generated by AI and based on your chart selections!")
                st.markdown(chart_insight)

                st.subheader("Congrats!")
                st.write("Now that you know the basics for interacting with Grant Scope lets move on to more advanced features.")
            if st.button("Continue", key="continue_button02"):
                st.session_state.intro_stage = 3
                st.experimental_rerun()

    elif st.session_state.intro_stage == 3:
        st.subheader("Unleash the Full Power of AI Analysis!")
        st.write("""
        You've done great so far! Now it's time to see what our AI can really do. 
        When you click the button below, we'll generate a comprehensive AI analysis of the entire dataset. 
        This might take a moment, but trust us, it's worth the wait!
        """)
        st.info("This is the last step in our introduction. Once you read through your data analysis dont forget to click the 'Complete' button to enable the full capabilities of GrantScope")

        if st.button("Generate Full AI Analysis", key="generate_full_ai_analysis_button03"):
            with st.spinner("Our AI is diving deep into the data... This might take a moment!"):
                full_analysis = generate_full_analysis(df, api_key)
            st.markdown(full_analysis)
            st.session_state.intro_stage = 4
            st.button("Complete the Introduction and open Dashboard", key="complete_the_introduction_and_open_dashboard_button04")

    elif st.session_state.intro_stage == 4:
        st.subheader("Congratulations! You're Ready to Explore!")
        st.write("""
        You've completed the introduction to GrantScope. You've seen how we can load data, 
        generate quick summaries, create custom charts, and perform deep AI analysis.

        Now you're ready to explore the full dashboard. You can:
        - Use the sidebar to navigate to different analysis pages
        - Choose a specific chart or analysis from the options below
        - Ask the AI your own questions about the data

        Remember, every page has AI-powered insights waiting for you. Happy exploring!
        """)

        # Add buttons or a dropdown for users to choose where to go next
        next_destination = st.selectbox("Where would you like to go next?",
                                        ["Data Summary", "Grant Amount Distribution", "Grant Amount Scatter Plot",
                                         "Grant Amount Heatmap", "Grant Description Word Clouds",
                                         "Treemaps with Extended Analysis"])

        if st.button("Go to Selected Page", key="go_to_selected_page_button05"):
            st.session_state.page = next_destination
            st.experimental_rerun()

    st.markdown("""
    ---
    GrantScope was created with ❤️ by [Your Name/Company]. We use the latest in AI technology, 
    including GPT-4 and LlamaIndex, to bring you powerful insights. Our colorful charts are made 
    with Plotly, and this whole interactive experience is powered by Streamlit. 

    Remember, while our AI is super smart, it's always learning. If you notice anything odd, 
    let us know - you might be teaching the AI something new!
    """)
