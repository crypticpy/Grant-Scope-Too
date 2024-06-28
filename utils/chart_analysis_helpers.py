from typing import Dict, Any

def generate_chart_specific_prompt(chart_type: str, chart_data: Dict[str, Any], project_theme: str = None) -> str:
    base_prompt = f"Analyze the {chart_type} chart for grant data. "
    
    if project_theme:
        base_prompt += f"Consider the project theme: {project_theme}. "
    
    if chart_type == "data_summary_metrics":
        prompt = base_prompt + f"""
        Analyze the summary metrics for the grant data.
        Focus on the following aspects:
        1. The overall scale of grant-making represented by these metrics.
        2. The diversity of the funding landscape as indicated by the number of unique funders and recipients.
        3. The implications of these metrics for grant seekers.

        Key data points to consider:
        - Total grants: {chart_data['total_grants']}
        - Total amount: ${chart_data['total_amount']:,.2f}
        - Unique funders: {chart_data['unique_funders']}
        - Unique recipients: {chart_data['unique_recipients']}
        """

    elif chart_type == "top_funders_chart":
        prompt = base_prompt + f"""
        Analyze the bar chart of top funders by total grant amount.
        Focus on the following aspects:
        1. The concentration of funding among the top funders.
        2. Any notable differences between the top funders and those lower on the list.
        3. Implications of this funding distribution for grant seekers.

        Key data points to consider:
        - Number of top funders shown: {chart_data['top_n']}
        - Top funder: {chart_data['top_funder']} (${chart_data['top_funder_amount']:,.2f})
        - Bottom funder in top {chart_data['top_n']}: {chart_data['bottom_top_funder']} (${chart_data['bottom_top_funder_amount']:,.2f})
        - Percentage of total funding from top {chart_data['top_n']} funders: {chart_data['top_n_percentage']:.2f}%
        """

    elif chart_type == "funder_type_pie_chart":
        prompt = base_prompt + f"""
        Analyze the pie chart showing the distribution of grants by funder type.
        Focus on the following aspects:
        1. The relative sizes of different funder types.
        2. Any surprising or notable aspects of this distribution.
        3. Implications of this distribution for grant seekers.

        Key data points to consider:
        - Total funder types: {chart_data['total_funder_types']}
        - Largest funder type: {chart_data['largest_funder_type']} ({chart_data['largest_funder_type_percentage']:.2f}%)
        - Smallest funder type: {chart_data['smallest_funder_type']} ({chart_data['smallest_funder_type_percentage']:.2f}%)
        """

    elif chart_type == "subject_area_chart":
        prompt = base_prompt + f"""
        Analyze the bar chart of top grant subject areas by total amount.
        Focus on the following aspects:
        1. The most prominent subject areas and their relative funding levels.
        2. Any notable patterns or surprises in the distribution of funding across subject areas.
        3. How this distribution might align with or differ from current trends or needs in philanthropy.

        Key data points to consider:
        - Number of subject areas shown: {chart_data['num_subject_areas']}
        - Top subject area: {chart_data['top_subject']} (${chart_data['top_subject_amount']:,.2f})
        - Bottom subject area shown: {chart_data['bottom_subject']} (${chart_data['bottom_subject_amount']:,.2f})
        - Percentage of total funding in top {chart_data['num_subject_areas']} areas: {chart_data['top_subjects_percentage']:.2f}%
        """

    elif chart_type == "time_series_chart":
        prompt = base_prompt + f"""
        Analyze the line chart showing total grant amount over time.
        Focus on the following aspects:
        1. The overall trend in grant amounts over the time period.
        2. Any notable spikes, dips, or changes in the trend.
        3. Potential factors that might explain the observed trends.

        Key data points to consider:
        - Time range: {chart_data['start_year']} to {chart_data['end_year']}
        - Lowest year: {chart_data['lowest_year']} (${chart_data['lowest_year_amount']:,.2f})
        - Highest year: {chart_data['highest_year']} (${chart_data['highest_year_amount']:,.2f})
        - Overall trend description: {chart_data['trend_description']}
        """

    elif chart_type == "network_graph":
        prompt = base_prompt + f"""
        Analyze the network graph of funders and recipients.
        Focus on the following aspects:
        1. The overall structure of the network and what it reveals about funder-recipient relationships.
        2. Any notable clusters or isolated nodes in the network.
        3. Implications of this network structure for grant seekers.

        Key data points to consider:
        - Total nodes (funders + recipients): {chart_data['total_nodes']}
        - Total connections: {chart_data['total_connections']}
        - Most connected node: {chart_data['most_connected_node']} ({chart_data['most_connected_node_connections']} connections)
        - Number of isolated nodes: {chart_data['isolated_nodes']}
        """

    prompt += """
    Provide insights, trends, and implications based ONLY on the data provided. 
    Do not give generic advice that isn't supported by the data.
    If the project theme is provided, relate your analysis to the theme where relevant.
    """
    
    return prompt