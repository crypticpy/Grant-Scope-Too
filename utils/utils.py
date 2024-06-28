import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import pdfkit
import streamlit as st
from streamlit.components.v1 import iframe

def clean_label(label: str) -> str:
    """
    Clean up labels by replacing underscores with spaces and capitalizing words.

    Args:
        label (str): The original label string.

    Returns:
        str: The cleaned up label.
    """
    return ' '.join(word.capitalize() for word in label.split('_'))


def download_excel(dataframes: List[pd.DataFrame], sheet_names: List[str], filename: str) -> None:
    """
    Create a download button for multiple dataframes as an Excel file with multiple sheets.

    Args:
        dataframes (List[pd.DataFrame]): List of pandas DataFrames to save.
        sheet_names (List[str]): List of sheet names (same length as dataframes).
        filename (str): Name of the Excel file to be saved.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)


def download_csv(df: pd.DataFrame, filename: str) -> str:
    """
    Generate a download link for a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved as CSV.
        filename (str): The name of the file to be downloaded.

    Returns:
        str: HTML string for download link.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


def generate_page_prompt(df: pd.DataFrame, grouped_df: pd.DataFrame, selected_chart: str, selected_role: str,
                         additional_context: str) -> str:
    columns = ', '.join(df.columns)
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    data_type_info = ", ".join([f"{col}: {dtype}" for col, dtype in data_types.items()])

    num_records = len(df)
    num_funders = df['Funder Name'].nunique()
    num_recipients = df['Recip Name'].nunique()
    observations = f"The dataset contains {num_records} records, with {num_funders} unique funders and {num_recipients} unique recipients."

    min_date = df['Year Issued'].min()
    max_date = df['Year Issued'].max()
    date_info = f"The dataset covers grants issued from {min_date} to {max_date}."

    unique_states = df['Funder State'].unique().tolist()
    geographical_info = f"The dataset covers grants from {len(unique_states)} states in the USA. The top states by grant count are {', '.join(df['Funder State'].value_counts().nlargest(3).index.tolist())}."

    total_amount = df['Amount Usd'].sum()
    avg_amount = df['Amount Usd'].mean()
    median_amount = df['Amount Usd'].median()
    aggregated_stats = f"The total grant amount is ${total_amount:,.2f}, with an average grant amount of ${avg_amount:,.2f} and a median grant amount of ${median_amount:,.2f}."

    top_subjects = df['Grant Subject Tran'].value_counts().nlargest(5).index.tolist()
    subject_info = f"The top 5 grant subjects are: {', '.join(top_subjects)}."

    top_populations = df['Grant Population Tran'].value_counts().nlargest(5).index.tolist()
    population_info = f"The top 5 populations served are: {', '.join(top_populations)}."

    top_strategies = df['Grant Strategy Tran'].value_counts().nlargest(5).index.tolist()
    strategy_info = f"The top 5 grant strategies are: {', '.join(top_strategies)}."

    funder_types = df['Funder Type'].unique().tolist()
    funder_type_info = f"The dataset includes the following funder types: {', '.join(funder_types)}."

    chart_description = f"The current chart is a {selected_chart}, which visualizes the grant data based on {additional_context}."
    role_description = f"The user is a {selected_role} who is exploring the grant data to gain insights and inform their work."

    prompt = f"""
    The Candid API provides comprehensive data on grants and funding in the USA. The current dataset contains the following columns: {columns}.
    Data types: {data_type_info}.
    You are an AI assistant helping a {selected_role} explore the grant data in the GrantScope application to gain insights and extract data useful to the grant application and writing process.
    {observations}
    {date_info}
    {geographical_info}
    {aggregated_stats}
    {subject_info}
    {population_info}
    {strategy_info}
    {funder_type_info}
    {chart_description}
    {role_description}
    The user can ask questions related to the current chart and the overall grant data to gain insights and explore the data further.
    Please note that the data is limited to the information provided in the dataset, queries beyond the available columns are not answerable.
    Respond in Markdown format only.
    The user's prompt is:
    """

    return prompt


def generate_dynamic_context(
        df: pd.DataFrame,
        page_name: str,
        selected_chart: str,
        selected_filters: Dict[str, Any],
        user_interactions: List[str],
        additional_context: Optional[Dict[str, Any]] = None,
        custom_context: Optional[Dict[str, Any]] = None
) -> str:
    context = f"The user is currently on the {page_name} page, viewing the {selected_chart} chart. "
    
    # Add information about the current dataset
    context += f"The dataset contains {len(df)} records from {df['Year Issued'].min()} to {df['Year Issued'].max()}. "
    
    if custom_context:
        if 'project_theme' in custom_context:
            context += f"Project Theme: {custom_context['project_theme']}. "
        if 'analysis_type' in custom_context:
            context += f"Current Analysis: {custom_context['analysis_type']}. "
    
    # Add filter information
    if selected_filters:
        context += "The following filters are applied: "
        context += ", ".join([f"{k}: {v}" for k, v in selected_filters.items()])
        context += ". "

    # Add recent user interactions
    if user_interactions:
        context += "Recent user interactions include: "
        context += ", ".join(user_interactions[-5:])
        context += ". "

    # Add additional context
    if additional_context:
        for key, value in additional_context.items():
            context += f"{key}: {value}. "

    return context


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, returning 0 if the denominator is 0.

    Args:
        numerator (float): The number to be divided.
        denominator (float): The number to divide by.

    Returns:
        float: The result of the division, or 0 if the denominator is 0.
    """
    return numerator / denominator if denominator != 0 else 0


def format_currency(amount: float) -> str:
    """
    Format a number as a currency string.

    Args:
        amount (float): The amount to be formatted.

    Returns:
        str: The formatted currency string.
    """
    return f"${amount:,.2f}"


def handle_error(func):
    """
    A decorator to handle exceptions in Streamlit apps.

    Args:
        func: The function to be wrapped.

    Returns:
        The wrapper function.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")

    return wrapper


def log_interaction(interaction: str):
    """
    Log a user interaction to the session state.

    Args:
        interaction (str): The interaction to log.
    """
    if 'user_interactions' not in st.session_state:
        st.session_state.user_interactions = []

    st.session_state.user_interactions.append(interaction)
    if len(st.session_state.user_interactions) > 20:  # Keep last 20 interactions
        st.session_state.user_interactions.pop(0)

def update_year_axis(fig, df, year_column='Year Issued'):
    unique_years = sorted(df[year_column].unique())
    fig.update_xaxes(
        tickvals=pd.to_datetime(unique_years, format='%Y'),
        ticktext=unique_years
    )
    return fig

def generate_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate a mapping of expected column names to actual column names in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, str]: A dictionary mapping expected column names to actual column names.
    """
    expected_columns = {
        'Grant Key': ['Grant Key', 'grant_key', 'Grant_Key'],
        'Amount Usd': ['Amount Usd', 'amount_usd', 'Amount_Usd'],
        'Funder Name': ['Funder Name', 'funder_name', 'Funder_Name'],
        'Recip Name': ['Recip Name', 'recip_name', 'Recip_Name'],
        'Funder Type': ['Funder Type', 'funder_type', 'Funder_Type'],
        'Grant Subject Tran': ['Grant Subject Tran', 'grant_subject_tran', 'Grant_Subject_Tran'],
        'Year Issued': ['Year Issued', 'year_issued', 'Year_Issued'],
        'Grant Description': ['Grant Description', 'grant_description', 'Grant_Description']
    }

    column_mapping = {}
    for expected_name, possible_names in expected_columns.items():
        for name in possible_names:
            if name in df.columns:
                column_mapping[expected_name] = name
                break
        if expected_name not in column_mapping:
            raise ValueError(f"Could not find a matching column for {expected_name}")

    return column_mapping


import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def create_pdf_report(content: str, filename: str) -> io.BytesIO:
    """
    Create a PDF report from the given content.

    Args:
        content (str): The content to be included in the PDF.
        filename (str): The name of the PDF file.

    Returns:
        io.BytesIO: A byte stream containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    flowables = []
    for line in content.split('\n'):
        if line.startswith('# '):
            flowables.append(Paragraph(line[2:], styles['Heading1']))
        elif line.startswith('## '):
            flowables.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('### '):
            flowables.append(Paragraph(line[4:], styles['Heading3']))
        else:
            flowables.append(Paragraph(line, styles['BodyText']))
        flowables.append(Spacer(1, 12))

    doc.build(flowables)
    buffer.seek(0)
    return buffer


def convert_streamlit_to_pdf():
    """
    Convert the current Streamlit page to a PDF and provide a download link.
    """
    # Get the HTML content of the current page
    html = """
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
            </style>
        </head>
        <body>
            %s
        </body>
    </html>
    """ % st._get_docstring()  # This gets the current page content

    # Convert HTML to PDF
    pdf = pdfkit.from_string(html, False)

    # Create a download link
    b64 = base64.b64encode(pdf).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'

    return href


def convert_streamlit_to_pdf_buffer(html_content):
    """
    Convert the given HTML content to a PDF and return it as a buffer.
    """
    # Convert HTML to PDF
    pdf_options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
    }
    pdf = pdfkit.from_string(html_content, False, options=pdf_options)

    return io.BytesIO(pdf)
