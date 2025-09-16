### Imports --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import chardet

from rfm_mapper_lib import (
    calculate_rfm_scores, 
    create_rfm_segments, 
    get_llm_cluster_names,
    get_llm_column_suggestions
)

from graphs import (
    create_cluster_size_plot,
    create_segment_scatterplot,
)


### Streamlit App Configuration -------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Intelligent Customer Segmentation")
st.write("This tool helps you segment your customers based on their purchasing behavior using Recency(last purchased date), Frequency(Number of times purchased), and Monetary(Total amount spent),  analysis combined with AI-driven insights.")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'rfm_df' not in st.session_state: 
    st.session_state.rfm_df = None
if 'rfm_with_clusters' not in st.session_state:
    st.session_state.rfm_with_clusters = None
if 'suggested_cols' not in st.session_state:
    st.session_state.suggested_cols = {}
if 'n_clusters_used' not in st.session_state:
    st.session_state.n_clusters_used = None
if 'llm_segments' not in st.session_state:
    st.session_state.llm_segments = None
if 'sales_trend_fig' not in st.session_state:
    st.session_state.sales_trend_fig = None
if 'clusters_df' not in st.session_state:
    st.session_state.clusters_df = None
if 'is_cluster_analysis_run' not in st.session_state:
    st.session_state.is_cluster_analysis_run = False
if 'segment_order' not in st.session_state:
    st.session_state.segment_order = []
if 'final_rfm_segments' not in st.session_state:
    st.session_state.final_rfm_segments = None


### Sidebar for file upload----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Upload Your Data")
    st.info("Please upload a CSV file containing your sales data. Ensure that the file includes columns for purchase date, customer ID, and transaction amount.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Check if a new file is uploaded
            if uploaded_file.name != st.session_state.get('last_uploaded_file'):
                rawdata = uploaded_file.read()
                result = chardet.detect(rawdata)
                encoding = result["encoding"]

                uploaded_file.seek(0)
                st.session_state.df = pd.read_csv(uploaded_file, encoding=encoding)
                
                with st.spinner("Uploading the dataset and analyzing columns..."):
                    st.session_state.suggested_cols = get_llm_column_suggestions(list(st.session_state.df.columns))
                
                st.session_state.last_uploaded_file = uploaded_file.name
                # Reset all analysis variables on new file upload
                st.session_state.rfm_df = None
                st.session_state.rfm_with_clusters = None 
                st.session_state.clusters_df = None
                st.session_state.llm_segments = None 
                st.session_state.sales_trend_fig = None 
                st.session_state.is_cluster_analysis_run = False
                st.session_state.segment_order = []
                st.session_state.final_rfm_segments = None
                st.success("File uploaded and AI analysis complete!")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
            st.session_state.suggested_cols = {}
            st.session_state.sales_trend_fig = None


# Main content area
if st.session_state.df is not None and uploaded_file is not None:
    st.write("Preview of uploaded data:")
    st.data_editor(st.session_state.df.head(), hide_index=True, use_container_width=True, disabled=True)
    st.divider()

    # --- Section 1: RFM Mapping & Calculation ---
    st.header("Column Mapping & Calculation")
    with st.expander("RFM Column Mapping", expanded=True):
        st.write("Please select the appropriate columns from your dataset for Recency, Frequency, and Monetary values. AI suggestions are provided to assist you.")
        
        col_recency, col_recency_ai = st.columns([0.7, 0.3], vertical_alignment="bottom")
        with col_recency:
            recency_col = st.selectbox(
                "Select Recency Column:",
                options=list(st.session_state.df.columns),
                index=list(st.session_state.df.columns).index(
                    st.session_state.suggested_cols.get('recency_col', '')
                ) if st.session_state.suggested_cols.get('recency_col', '') in list(st.session_state.df.columns) else 0,
                key='recency_col',
                help="This column should contain date or timestamp values representing when the purchase was made."
            )
        with col_recency_ai:
            st.info(f"AI Suggestion: **`{st.session_state.suggested_cols.get('recency_col', 'N/A')}`**", icon="🤖")

        col_frequency, col_frequency_ai = st.columns([0.7, 0.3], vertical_alignment="bottom")
        with col_frequency:
            frequency_col = st.selectbox(
                "Select Frequency Column:",
                options=list(st.session_state.df.columns),
                index=list(st.session_state.df.columns).index(st.session_state.suggested_cols.get('frequency_col', '')) if st.session_state.suggested_cols.get('frequency_col', '') in list(st.session_state.df.columns) else 0,
                key='frequency_col',
                help="This column should contain unique identifiers for each customer."
            )
        with col_frequency_ai:
            st.info(f"AI Suggestion: **`{st.session_state.suggested_cols.get('frequency_col', 'N/A')}`**", icon="🤖")

        col_monetary, col_monetary_ai = st.columns([0.7, 0.3], vertical_alignment="bottom")
        with col_monetary:
            monetary_col = st.selectbox(
                "Select Monetary Column:",
                options=list(st.session_state.df.columns),
                index=list(st.session_state.df.columns).index(st.session_state.suggested_cols.get('monetary_col', '')) if st.session_state.suggested_cols.get('monetary_col', '') in list(st.session_state.df.columns) else 0,
                key='monetary_col',
                help="This column should contain numerical values representing the amount spent in each transaction."
            )
        with col_monetary_ai:
            st.info(f"AI Suggestion: **`{st.session_state.suggested_cols.get('monetary_col', 'N/A')}`**", icon="🤖")
        
    if st.button("Calculate Metrics"):
        if recency_col and frequency_col and monetary_col:
            with st.spinner("Calculating scores..."):
                st.session_state.rfm_df = calculate_rfm_scores(st.session_state.df, recency_col, frequency_col, monetary_col)
            if st.session_state.rfm_df is not None:
                st.success("RFM Metrics calculated successfully!")
            else:
                st.error("Failed to calculate RFM values. Please check your column selections.") 
        else:
            st.warning("Please select all three required columns.")
    
    # --- Section 2: Cluster Calculation & Personalization ---
    st.divider()
    
    if st.session_state.rfm_df is not None:
        with st.container(border=True):

            st.header("Cluster Analysis")
            col_calc, col_personalize = st.columns(2)
            with col_calc:
                st.subheader("Run Optimal Clustering")
                st.write("Automatically finds an optimal number of clusters")
                if st.button("Run Optimal Cluster Analysis"):
                    with st.spinner("Finding optimal clusters and segmenting customers..."):
                        st.session_state.rfm_with_clusters, st.session_state.clusters_df, st.session_state.n_clusters_used = create_rfm_segments(st.session_state.rfm_df)
                        st.write(f"Optimal number of clusters determined: **{st.session_state.n_clusters_used}**")
                    
                    with st.spinner("Getting segment descriptions from LLM..."):
                        cluster_centers_json = st.session_state.clusters_df.to_json(orient='records')
                        st.session_state.llm_segments = get_llm_cluster_names(cluster_centers_json)
                        if st.session_state.llm_segments:
                            segment_list = st.session_state.llm_segments.get("segments", [])
                            sort_order = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising", "About to Sleep", "At-Risk", "Can't Lose Them", "Hibernating", "Lost"]
                            sort_keys = {name: i for i, name in enumerate(sort_order)}
                            sorted_segments = sorted(segment_list, key=lambda x: sort_keys.get(x.get('segment_name', ''), len(sort_order)))
                            st.session_state.llm_segments["segments"] = sorted_segments
                            st.session_state.segment_order = [s.get('segment_name') for s in sorted_segments]
                            
                    st.session_state.is_cluster_analysis_run = True
                    st.success("Clustering and Segment Naming complete!")
                    st.rerun()
            
            with col_personalize:
                st.subheader("Run Custom Clustering")
                st.write("Specify a custom number of clusters to analyze.")
                user_n_clusters = st.number_input(
                    "Enter desired number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=st.session_state.n_clusters_used if st.session_state.n_clusters_used else 5,
                    step=1,
                    key='user_n_clusters_input'
                )
                if st.button("Run Custom Cluster Analysis"):
                    with st.spinner(f"Re-running analysis with {user_n_clusters} clusters..."):
                        st.session_state.rfm_with_clusters, st.session_state.clusters_df, st.session_state.n_clusters_used = create_rfm_segments(st.session_state.rfm_df, n_clusters=user_n_clusters)
                    
                    with st.spinner("Getting segment descriptions from LLM..."):
                        cluster_centers_json = st.session_state.clusters_df.to_json(orient='records')
                        st.session_state.llm_segments = get_llm_cluster_names(cluster_centers_json)
                        if st.session_state.llm_segments:
                            segment_list = st.session_state.llm_segments.get("segments", [])
                            sort_order = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "Promising", "About to Sleep", "At-Risk", "Can't Lose Them", "Hibernating", "Lost"]
                            sort_keys = {name: i for i, name in enumerate(sort_order)}
                            sorted_segments = sorted(segment_list, key=lambda x: sort_keys.get(x.get('segment_name', ''), len(sort_order)))
                            st.session_state.llm_segments["segments"] = sorted_segments
                            st.session_state.segment_order = [s.get('segment_name') for s in sorted_segments]
                            
                    st.session_state.is_cluster_analysis_run = True
                    st.success("Analysis complete!")
                    st.rerun()

    # --- Section 3: Segment Descriptions & Data Download ---
    st.divider()
    if st.session_state.is_cluster_analysis_run:

        with st.container(border=True):
            st.header("Segment Descriptions & Recommendations")
            st.write("Based on the cluster analysis, here are the names and recommendations for your customer segments:")
            if st.session_state.llm_segments:
                segments_list = st.session_state.llm_segments.get("segments", [])
                df_segments = pd.DataFrame(segments_list)
                df_segments.rename(columns={'cluster_id': 'Cluster',
                                            'segment_name': 'Segment Name',
                                            'description': 'Segment Description',
                                            'recommendations': 'Marketing Recommendations'}, inplace=True)
                
                df_segments['Segment Name'] = pd.Categorical(df_segments['Segment Name'], categories=st.session_state.segment_order, ordered=True)
                df_segments.sort_values('Segment Name', inplace=True)
                df_segments.reset_index(drop=True, inplace=True)
                if not df_segments.empty:
                    df_segments.insert(0, "Sr No.", range(1, len(df_segments) + 1))

                    # Drop Cluster column if not needed
                    df_segments = df_segments.drop(columns=['Cluster'])

                    # Add CSS once
                    st.markdown("""
                    <style>
                    .custom-table {
                    border-collapse: separate;
                    border-spacing: 0;
                    width: 100%;
                    border: 1px solid #000;
                    border-radius: 10px;
                    overflow: hidden; /* clip rounded corners */
                    }

                    .custom-table th, .custom-table td {
                    padding: 8px 12px;
                    text-align: left;  /* align header and cell text to left */
                    white-space: normal;
                    word-wrap: break-word;
                    vertical-align: top;
                    border-bottom: 1px solid #eee;
                    }

                    .custom-table th {
                    background-color: #000;
                    font-weight: 600;
                    }

                    .custom-table tr:nth-child(even) td {
                    background-color: #262730; /* zebra stripe */
                    }

                    .custom-table tr:hover td {
                    background-color: #c3d5fa; /* light highlight on hover */
                    color: #000;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Render dataframe as styled HTML
                    st.markdown(
                        df_segments.to_html(escape=False, index=False, classes="custom-table"),
                        unsafe_allow_html=True
                    )



                    # st.markdown(
                    #     df_segments.to_html(escape=False, index=False).replace(
                    #         '<td', '<td style="white-space: normal; word-wrap: break-word;"'
                    #     ),
                    #     unsafe_allow_html=True
                    # )
                
                    
                    
                    
                    
                    
                else:
                    st.info("No segment descriptions available.")
            else:
                st.error("Failed to get segment descriptions from the LLM.")
        
        # with st.container(border=True):
        #     st.write("Preview and download the final dataset with cluster assignments and segment names.")
        if st.session_state.final_rfm_segments is None: # Prepare dataframe only once
            segments_df = pd.DataFrame(st.session_state.llm_segments.get('segments', [])).rename(columns={'cluster_id': 'Cluster', 'segment_name': 'Segment Name'})
            final_rfm = st.session_state.rfm_with_clusters.reset_index().merge(segments_df[['Cluster', 'Segment Name']], on='Cluster', how='left')
            st.session_state.final_rfm_segments = final_rfm[[col for col in final_rfm.columns if col not in ['Segment Name', 'Cluster']] + ['Cluster', 'Segment Name']]

        #     st.dataframe(st.session_state.final_rfm_segments.head(10), use_container_width=True, hide_index=True)
            
        #     st.download_button(
        #         label="Download RFM Segments CSV",
        #         data=st.session_state.final_rfm_segments.to_csv(index=False).encode('utf-8'),
        #         file_name='rfm_segments.csv',
        #         mime='text/csv',
        #     )

    # --- Section 4: Cluster Visualization ---
    st.divider()
    if st.session_state.is_cluster_analysis_run and st.session_state.final_rfm_segments is not None:
        with st.container(border=True):
            st.header("Segmentation Visualization")
            st.write("Explore your customer segments using the interactive visualizations below.")
            
            # tab_list = [
            #     "📊 Segment Overview", "📈 2D Scatterplot", "☀️ Sunburst",
            #     "🌐 3D Cluster Plot"
            # ]
            # tab1, tab2, tab3, tab4 = st.tabs(tab_list)
            
            
            tab_list = [
                "📊 Segment Size", "📈 2D Segment Plot"]
            tab1, tab2 = st.tabs(tab_list)
            

            with tab1:
                st.subheader("Segment Size")
                st.plotly_chart(create_cluster_size_plot(st.session_state.final_rfm_segments), use_container_width=True)

            with tab2:
                st.subheader("2D Segment Plot")
                st.plotly_chart(create_segment_scatterplot(st.session_state.clusters_df, st.session_state.llm_segments), use_container_width=True)

            # with tab3:
            #     st.subheader("Interactive Sunburst Chart")
            #     st.plotly_chart(create_sunburst_plot(st.session_state.final_rfm_segments), use_container_width=True)

            # with tab4:
            #     st.subheader("3D RFM Cluster Visualization")
            #     st.plotly_chart(create_3d_cluster_plot(st.session_state.clusters_df, st.session_state.rfm_with_clusters, st.session_state.llm_segments), use_container_width=True)
            



    # --- Section 5: Individual Customer Sales Trend ---
    # st.divider()
    # st.header("Individual Customer Sales Trend")
    # if st.session_state.rfm_df is not None:
    #     with st.container(border=True):
    #         st.write("Select a customer from the dropdown to visualize their sales trend over time.")

    #         frequency_col_name = st.session_state.suggested_cols.get('frequency_col')
    #         recency_col = st.session_state.get('recency_col')
    #         monetary_col = st.session_state.get('monetary_col')
            
    #         if frequency_col_name in st.session_state.df.columns:
    #             customer_ids = st.session_state.df[frequency_col_name].unique()
    #             selected_customer = st.selectbox("Select a Customer ID:", options=customer_ids, key='customer_id_select')

    #             if st.button("Generate Sales Trend Graph"):
    #                 with st.spinner("Generating graph..."):
    #                     fig_trend = create_customer_sales_trend_graph(
    #                         st.session_state.df,
    #                         selected_customer,
    #                         recency_col,
    #                         monetary_col,
    #                         frequency_col_name
    #                     )
    #                     st.session_state.sales_trend_fig = fig_trend
                        
    #                 if st.session_state.sales_trend_fig is None:
    #                     st.info("Not enough data to plot a trend for the selected customer.")

    #         if st.session_state.sales_trend_fig is not None:
    #             st.plotly_chart(st.session_state.sales_trend_fig, use_container_width=True)

    # # --- Section 6: Custom Chart Builder ---
    # st.divider()
    # st.header("Custom Chart Builder")
    # with st.container(border=True):
    #     st.write("Create your own visualizations from the original dataset.")
        
    #     chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram"])

    #     try:
    #         if chart_type in ["Bar", "Line", "Scatter", "Pie"]:
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 x_axis = st.selectbox("Select X-Axis", options=[None] + list(st.session_state.df.columns), key="custom_x")
    #             with col2:
    #                 y_axis = st.selectbox("Select Y-Axis", options=[None] + list(st.session_state.df.columns), key="custom_y")
                
    #             color_axis = st.selectbox("Select Color Dimension (Optional)", options=[None] + list(st.session_state.df.columns), key="custom_color")

    #             if x_axis and y_axis:
    #                 if chart_type == "Bar":
    #                     fig = px.bar(st.session_state.df, x=x_axis, y=y_axis, color=color_axis, title=f"{x_axis} vs {y_axis}")
    #                 elif chart_type == "Line":
    #                     fig = px.line(st.session_state.df, x=x_axis, y=y_axis, color=color_axis, title=f"{x_axis} vs {y_axis}")
    #                 elif chart_type == "Scatter":
    #                     fig = px.scatter(st.session_state.df, x=x_axis, y=y_axis, color=color_axis, title=f"{x_axis} vs {y_axis}")
    #                 elif chart_type == "Pie":
    #                     fig = px.pie(st.session_state.df, names=x_axis, values=y_axis, color=color_axis, title=f"Distribution of {y_axis} by {x_axis}")
                    
    #                 st.plotly_chart(fig, use_container_width=True)

    #         elif chart_type == "Histogram":
    #             hist_col = st.selectbox("Select Column for Histogram", options=[None] + list(st.session_state.df.columns), key="custom_hist")
    #             if hist_col:
    #                 fig = px.histogram(st.session_state.df, x=hist_col, title=f"Distribution of {hist_col}")
    #                 st.plotly_chart(fig, use_container_width=True)

    #     except Exception as e:
    #         st.error(f"Could not generate chart. Please check column selections. Error: {e}")

