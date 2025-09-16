import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np



def create_cluster_size_plot(rfm_df):
    # Define sort order for segments
    sort_order = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising", "About to Sleep", "At-Risk Customers", "Can't Lose Them", "Hibernating", "Lost Customers"]
    
    # Count customers per segment
    cluster_counts = rfm_df['Segment Name'].value_counts().reset_index()
    cluster_counts.columns = ['Segment Name', 'Count']
    
    # Apply categorical sorting (ignore missing categories gracefully)
    cluster_counts['Segment Name'] = pd.Categorical(
        cluster_counts['Segment Name'],
        categories=sort_order,
        ordered=True
    )
    
    cluster_counts = cluster_counts.sort_values('Segment Name')
    
    # Create bar chart
    fig = px.bar(
        cluster_counts,
        x='Segment Name',
        y='Count',
        color='Segment Name',
        title='Number of Customers in Each Segment'
    )
    
    # Optional: Pie chart (uncomment if needed)
    # fig = px.pie(
    #     cluster_counts,
    #     names='Segment Name',
    #     values='Count',
    #     title='Customer Distribution by Segment'
    # )
    
    return fig


def create_segment_scatterplot(clusters_df, llm_segments):
    """
    Creates a scatterplot of the RFM segments based on mean values.
    The x-axis is Recency (reversed), y-axis is Monetary, and the size of the points is Frequency.
    The plot includes segment names as labels and quadrant lines.
    """
    # Get segment names from LLM segments data and merge with clusters_df
    segments_df = pd.DataFrame(llm_segments.get('segments', [])).rename(columns={'cluster_id': 'Cluster', 'segment_name': 'Segment Name'})
    merged_df = clusters_df.merge(segments_df[['Cluster', 'Segment Name']], on='Cluster', how='left')
    
    # Scale Frequency for marker size
    size_min = 30
    size_max = 120
    freq_min = merged_df['Frequency'].min()
    freq_max = merged_df['Frequency'].max()
    
    # Avoid division by zero if all frequencies are the same
    if freq_max > freq_min:
        sizes = size_min + (merged_df['Frequency'] - freq_min) * (size_max - size_min) / (freq_max - freq_min)
    else:
        sizes = np.full(merged_df['Frequency'].shape, size_min)

    fig = go.Figure()

    # Add the scatter plot for the segments
    fig.add_trace(go.Scatter(
        x=merged_df['Recency'],
        y=merged_df['Monetary'],
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=merged_df['Cluster'],
            colorscale='Viridis',
            showscale=False,
            opacity=0.8,
            # line=dict(width=1, color='DarkSlateGrey')
            line=dict(width=1, color='white')

        ),
        text=merged_df['Segment Name'],
        textposition='bottom center',
        textfont=dict(
            size=12,
            color='white',
        ),
        hoverinfo='text',
        hovertext=[
            f"<b>Segment:</b> {row['Segment Name']}<br>"
            f"<b>Recency:</b> {row['Recency']:.2f}<br>"
            f"<b>Frequency:</b> {row['Frequency']:.2f}<br>"
            f"<b>Monetary:</b> {row['Monetary']:.2f}"
            for index, row in merged_df.iterrows()
        ]
    ))
    
    # Add quadrant lines and styling
    mean_recency = merged_df['Recency'].mean()
    mean_monetary = merged_df['Monetary'].mean()
    
    fig.add_shape(type="line",
                  x0=mean_recency, 
                  y0=merged_df['Monetary'].min(), 
                  x1=mean_recency, 
                  y1=merged_df['Monetary'].max(),
                  line=dict(color="Gray", width=2, dash="dash")
    )
    
    fig.add_shape(type="line",
                  x0=merged_df['Recency'].min(),
                  y0=mean_monetary,
                  x1=merged_df['Recency'].max(),
                  y1=mean_monetary,
                  line=dict(color="Gray", width=2, dash="dash")
    )

    # Adjust axis ranges dynamically to prevent crowding at the edges
    fig.update_layout(
        title='Segment Scatterplot',
        xaxis=dict(
            title='Frequency (Days)',
            autorange='reversed', # Reversing recency to show better segments on right
            showgrid=False,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Monetary  Value (USD)',
            autorange=True,
            showgrid=False,
            gridcolor='lightgray',
            zeroline=False
        ),
        showlegend=False,
        
    )
    
    return fig


