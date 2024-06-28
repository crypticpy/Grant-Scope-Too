import io
import json
import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

from utils.utils import clean_label, handle_error, safe_divide

@dataclass
class Grant:
    funder_key: str
    funder_profile_url: str
    funder_name: str
    funder_city: str
    funder_state: str
    funder_country: str
    funder_type: str
    funder_zipcode: str
    funder_country_code: str
    funder_ein: str
    funder_gs_profile_update_level: str
    recip_key: str
    recip_name: str
    recip_city: str
    recip_state: str
    recip_country: str
    recip_zipcode: str
    recip_country_code: str
    recip_ein: str
    recip_organization_code: str
    recip_organization_tran: str
    recip_gs_profile_link: str
    recip_gs_profile_update_level: str
    grant_key: str
    amount_usd: int
    grant_subject_code: str
    grant_subject_tran: str
    grant_population_code: str
    grant_population_tran: str
    grant_strategy_code: str
    grant_strategy_tran: str
    grant_transaction_code: str
    grant_transaction_tran: str
    grant_geo_area_code: str
    grant_geo_area_tran: str
    year_issued: str
    grant_duration: str
    grant_description: str
    last_updated: str

@dataclass
class Grants:
    grants: List[Grant]

@st.cache_data
@handle_error
def load_data(file_path: Optional[str] = None, uploaded_file: Optional[Any] = None) -> Optional[Grants]:
    """
    Load grant data from a file or uploaded file.

    Args:
        file_path (Optional[str]): Path to the JSON file.
        uploaded_file (Optional[Any]): Uploaded file object from Streamlit.

    Returns:
        Optional[Grants]: Grants object containing the loaded data, or None if loading fails.
    """
    if uploaded_file is not None:
        data = json.load(io.BytesIO(uploaded_file.read()))
    elif file_path is not None:
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        raise ValueError("Either file_path or uploaded_file must be provided.")

    grants = Grants(grants=[Grant(**grant) for grant in data['grants']])
    return grants

@st.cache_data
@handle_error
def preprocess_data(grants: Grants) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the grants data.

    Args:
        grants (Grants): Grants object containing the raw data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the processed DataFrame and the grouped DataFrame.
    """
    df = pd.DataFrame([asdict(grant) for grant in grants.grants])

    df['grant_index'] = df['grant_key']
    df['year_issued'] = pd.to_numeric(df['year_issued'], errors='coerce').fillna(0).astype(int)
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df = df.dropna(subset=['amount_usd'])

    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    code_columns = [col for col in df.columns if "_code" in col]
    tran_columns = [col for col in df.columns if "_tran" in col]

    for code_col, tran_col in zip(code_columns, tran_columns):
        df[[code_col, tran_col]] = df[[code_col, tran_col]].map(
            lambda x: x.split(';') if isinstance(x, str) else ['Unknown']
        )
        df = df.explode(code_col).explode(tran_col)

    df = df.fillna({'object': 'Unknown', 'number': df.select_dtypes(include=['number']).median()})
    df['grant_description'] = df['grant_description'].fillna('').astype(str)

    bins = [0, 50000, 100000, 500000, 1000000, np.inf]
    names = ['0-50k', '50k-100k', '100k-500k', '500k-1M', '1M+']
    df['amount_usd_cluster'] = pd.cut(df['amount_usd'], bins, labels=names)

    df = df.drop_duplicates(subset=['year_issued', 'grant_key'])

    grouped_df = df.groupby('grant_index').first()

    # Apply clean_label to all column names
    df.columns = [clean_label(col) for col in df.columns]
    grouped_df.columns = [clean_label(col) for col in grouped_df.columns]

    return df, grouped_df

def calculate_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the grants data.

    Args:
        df (pd.DataFrame): The processed DataFrame.

    Returns:
        dict: A dictionary containing summary statistics.
    """
    total_grants = len(df)
    total_amount = df['Amount Usd'].sum()
    avg_amount = safe_divide(total_amount, total_grants)
    unique_funders = df['Funder Name'].nunique()
    unique_recipients = df['Recip Name'].nunique()

    return {
        'Total Grants': total_grants,
        'Total Amount': total_amount,
        'Average Grant Amount': avg_amount,
        'Unique Funders': unique_funders,
        'Unique Recipients': unique_recipients
    }
