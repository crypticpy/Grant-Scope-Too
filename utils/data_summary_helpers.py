import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import networkx as nx
from typing import Dict
import streamlit as st

def create_summary_metrics(df: pd.DataFrame, column_mapping: Dict[str, str]) -> None:
    """Create and display summary metrics with larger text and no decimals."""
    metrics = [
        ("Total Grants", df[column_mapping['Grant Key']].nunique()),
        ("Total Amount", df[column_mapping['Amount Usd']].sum()),
        ("Unique Funders", df[column_mapping['Funder Name']].nunique()),
        ("Unique Recipients", df[column_mapping['Recip Name']].nunique())
    ]

    # CSS to increase text size
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    .medium-font {
        font-size:24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create two rows of metrics
    for i in range(0, len(metrics), 2):
        col1, col2 = st.columns(2)

        with col1:
            label, value = metrics[i]
            st.markdown(f"<p class='medium-font'>{label}</p>", unsafe_allow_html=True)
            if label == "Total Amount":
                st.markdown(f"<p class='big-font'>${format_large_number(value)}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='big-font'>{format_large_number(value)}</p>", unsafe_allow_html=True)

        if i + 1 < len(metrics):
            with col2:
                label, value = metrics[i + 1]
                st.markdown(f"<p class='medium-font'>{label}</p>", unsafe_allow_html=True)
                if label == "Total Amount":
                    st.markdown(f"<p class='big-font'>${format_large_number(value)}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='big-font'>{format_large_number(value)}</p>", unsafe_allow_html=True)

def create_top_funders_chart(df: pd.DataFrame, top_n: int, column_mapping: Dict[str, str]) -> go.Figure:
    """Create a bar chart of top funders."""
    unique_df = df.drop_duplicates(subset=column_mapping['Grant Key'])
    top_funders = unique_df.groupby(column_mapping['Funder Name'])[column_mapping['Amount Usd']].sum().nlargest(top_n).reset_index()

    fig = px.bar(top_funders,
                 x=column_mapping['Amount Usd'],
                 y=column_mapping['Funder Name'],
                 orientation='h',
                 title=f"Top {top_n} Funders by Total Grant Amount")

    fig.update_layout(
        xaxis_title="Total Grant Amount (USD)",
        yaxis_title="Funder Name",
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

def create_funder_type_pie_chart(df: pd.DataFrame, column_mapping: Dict[str, str]) -> go.Figure:
    """Create a pie chart of grant distribution by funder type."""
    unique_df = df.drop_duplicates(subset=column_mapping['Grant Key'])
    funder_type_dist = unique_df.groupby(column_mapping['Funder Type'])[column_mapping['Amount Usd']].sum().reset_index()

    fig = px.pie(funder_type_dist,
                 values=column_mapping['Amount Usd'],
                 names=column_mapping['Funder Type'],
                 title="Grant Distribution by Funder Type")

    return fig

def create_subject_area_chart(df: pd.DataFrame, column_mapping: Dict[str, str]) -> go.Figure:
    """Create a bar chart of top grant subject areas."""
    unique_df = df.drop_duplicates(subset=column_mapping['Grant Key'])
    subject_dist = unique_df.groupby(column_mapping['Grant Subject Tran'])[column_mapping['Amount Usd']].sum().nlargest(10).reset_index()

    fig = px.bar(subject_dist,
                 x=column_mapping['Grant Subject Tran'],
                 y=column_mapping['Amount Usd'],
                 title="Top 10 Grant Subject Areas by Total Amount")

    fig.update_layout(
        xaxis_title="Subject Area",
        yaxis_title="Total Grant Amount (USD)",
        xaxis={'categoryorder': 'total descending'}
    )

    return fig

def create_time_series_chart(df: pd.DataFrame, column_mapping: Dict[str, str]) -> go.Figure:
    """Create a line chart of total grant amount over time."""
    time_series = df.groupby(column_mapping['Year Issued'])[column_mapping['Amount Usd']].sum().reset_index()

    # Ensure 'Year Issued' is treated as a datetime
    time_series[column_mapping['Year Issued']] = pd.to_datetime(time_series[column_mapping['Year Issued']], format='%Y')

    fig = px.line(time_series,
                  x=column_mapping['Year Issued'],
                  y=column_mapping['Amount Usd'],
                  title="Total Grant Amount Over Time")

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Grant Amount (USD)",
        xaxis=dict(
            tickmode='array',
            tickvals=time_series[column_mapping['Year Issued']],
            ticktext=time_series[column_mapping['Year Issued']].dt.year,
            tickangle=45
        )
    )

    return fig

def create_interactive_network_graph(df: pd.DataFrame, column_mapping: Dict[str, str]) -> go.Figure:
    """Create an interactive network graph of funders and recipients."""
    G = nx.from_pandas_edgelist(df, column_mapping['Funder Name'], column_mapping['Recip Name'], column_mapping['Amount Usd'])

    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    edge_texts = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_texts.append(f"Funder: {edge[0]}<br>Recipient: {edge[1]}<br>Amount: ${edge[2][column_mapping['Amount Usd']]:,.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_texts
    )

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_adjacencies = []
    node_texts = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        if node in df[column_mapping['Funder Name']].values:
            total_granted = df[df[column_mapping['Funder Name']] == node][column_mapping['Amount Usd']].sum()
            node_texts.append(f"Funder: {node}<br>Total Granted: ${total_granted:,.2f}<br>Connections: {len(adjacencies)}")
        else:
            total_received = df[df[column_mapping['Recip Name']] == node][column_mapping['Amount Usd']].sum()
            node_texts.append(f"Recipient: {node}<br>Total Received: ${total_received:,.2f}<br>Connections: {len(adjacencies)}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        text=node_texts
    )

    node_trace.marker.color = node_adjacencies
    node_trace.marker.size = [min(20 + n * 2, 50) for n in node_adjacencies]

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Network of Funders and Recipients',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Node size represents number of connections.<br>Color intensity indicates connection strength.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig

def format_large_number(num: float) -> str:
    """Format numbers into a readable format with no decimals."""
    num = int(num)  # Convert to integer to remove any fractional part
    if num >= 1e9:
        return f"{num // 1e9}B"
    elif num >= 1e6:
        return f"{num // 1e6}M"
    elif num >= 1e3:
        return f"{num // 1e3}K"
    else:
        return f"{num}"