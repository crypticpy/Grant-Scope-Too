from typing import Dict, Any
import pandas as pd
from utils.utils import format_currency, clean_label
from loaders.llama_index_setup import query_data

from typing import Dict, Any
import pandas as pd
from utils.utils import format_currency, clean_label
from loaders.llama_index_setup import query_data

def generate_full_analysis(df: pd.DataFrame, api_key: str, project_theme: str) -> str:
    """
    Generate a comprehensive analysis of the grant data, considering the project theme.
    """
    # Prepare context with key statistics
    context = {
        "total_grants": len(df),
        "total_amount": format_currency(df['Amount Usd'].sum()),
        "average_grant": format_currency(df['Amount Usd'].mean()),
        "median_grant": format_currency(df['Amount Usd'].median()),
        "unique_funders": df['Funder Name'].nunique(),
        "unique_recipients": df['Recip Name'].nunique(),
        "year_range": f"{df['Year Issued'].min()} to {df['Year Issued'].max()}",
        "top_funder": df.groupby('Funder Name')['Amount Usd'].sum().nlargest(1).index[0],
        "top_funder_amount": format_currency(df.groupby('Funder Name')['Amount Usd'].sum().nlargest(1).values[0]),
        "top_subject": df['Grant Subject Tran'].value_counts().index[0],
        "top_population": df['Grant Population Tran'].value_counts().index[0],
        "top_strategy": df['Grant Strategy Tran'].value_counts().index[0],
    }

    # Create analysis prompt
    analysis_prompt = f"""
    Provide a comprehensive analysis of the grant dataset, considering the following project theme and objectives:

    {project_theme}

    Use the following structure and information:

    1. Overview:
       - Summarize the key statistics of the dataset.
       - Discuss the scope of the data in terms of total grants, amounts, and time period.

    2. Funding Landscape:
       - Analyze the distribution of grants across different funder types.
       - Discuss the concentration of funding (e.g., are there a few major funders or is it more distributed?).
       - Identify any trends in funding amounts over the years.

    3. Grant Characteristics:
       - Examine the most common grant subjects, target populations, and strategies.
       - Discuss how grant amounts vary across these categories.
       - Identify any notable patterns or relationships between grant characteristics and funding amounts.

    4. Temporal Analysis:
       - Analyze how grant-making has evolved over the covered time period.
       - Identify any significant changes or trends in funding priorities, amounts, or number of grants over time.

    5. Key Players:
       - Discuss the top funders and recipients in terms of grant amounts and numbers.
       - Analyze any patterns in the relationships between top funders and recipients.

    6. Relevance to Project Theme:
       - Highlight aspects of the data that are particularly relevant to the project theme and objectives.
       - Identify potential funding opportunities or strategies based on the data that align with the project goals.

    7. Implications and Recommendations:
       - Based on the analysis, provide insights on the overall state of grant-making in this dataset.
       - Suggest areas that might benefit from further investigation, considering the project theme.
       - Offer specific, data-driven recommendations for grant seekers based on the observed patterns and trends.

    Use the provided context and your analysis of the full dataset to support your points. 
    Focus on insights that can be derived from the data and are relevant to the user's project.
    Avoid giving generic advice that isn't supported by the data provided.
    Aim for a college-level analysis that is informative and insightful, but not overwhelmingly technical.
    """

    # Generate the analysis using LlamaIndex
    full_analysis = query_data(df, analysis_prompt, "Full grant data analysis", context, api_key)

    return full_analysis