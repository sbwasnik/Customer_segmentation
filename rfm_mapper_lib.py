import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import json
import requests
import streamlit as st
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import numpy as np


AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]


warnings.filterwarnings('ignore')

def get_llm_column_suggestions(column_names):
    """
    Calls an LLM to suggest appropriate RFM columns based on DataFrame headers.
    """
    try:
        # Define the system prompt
        system_prompt = "You are a highly intelligent data analyst specializing in customer segmentation. Your task is to analyze a list of column headers from a dataset and identify the best-suited columns for Recency, Frequency, and Monetary analysis. Your response must be a JSON object with keys 'recency_col', 'frequency_col', and 'monetary_col'. Only use the column names provided in the user query. Your top priority is to find a suitable column for all three RFM categories. If you cannot find a suitable column, indicate with 'N/A'."

        # Define the user prompt
        user_query = f"Here is a list of column headers from a sales dataset: {column_names}\n\n" \
                     "Please identify the most suitable column for each of the following:\n" \
                     "1. Recency (Purchase Date/Timestamp)\n" \
                     "2. Frequency (Customer ID/Identifier)\n" \
                     "3. Monetary (Transaction Amount/Value)\n\n" \
                     "Return a JSON object with the most likely column names."
        
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.5,
            "max_tokens": 400,
            "response_format": {"type": "json_object"}
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and result['choices']:
            json_content = result['choices'][0]['message']['content']
            return json.loads(json_content)
        

    except requests.exceptions.RequestException as e:
        st.error(f"Network or API request error: {e}")
        return {}
    except json.JSONDecodeError:
        st.error("LLM response was not a valid JSON object.")
        return {}
    
def calculate_rfm_scores(df, recency_col, frequency_col, monetary_col):
    """
    Calculates RFM values for each customer based on the mapped columns.
    """
    processed_df = df.copy()
    processed_df[recency_col] = pd.to_datetime(processed_df[recency_col], format='mixed', dayfirst=True, errors='coerce')

    unparsed_dates = processed_df[recency_col].isnull().sum()
    if unparsed_dates > 0:
        print(f"Could not parse {unparsed_dates} date(s) in the '{recency_col}' column. These rows will be excluded from the analysis.")

    # Drop rows with unparsable dates to prevent calculation errors.
    processed_df.dropna(subset=[recency_col], inplace=True)

    # Use the latest date in the dataset for Recency calculation.
    latest_date = processed_df[recency_col].max()
    processed_df['Recency'] = (latest_date - processed_df[recency_col]).dt.days

    # Calculate Frequency and Monetary
    rfm_df = processed_df.groupby(frequency_col).agg(
        Frequency=(frequency_col, 'count'),
        Monetary=(monetary_col, 'sum'),
        Recency=('Recency', 'min')
    ).reset_index()

    # Set the customer ID as the index
    rfm_df = rfm_df.set_index(frequency_col)
    return rfm_df

def find_optimal_clusters(data_scaled, min_clusters=4, max_clusters=7):
    """
    Determines the optimal number of clusters using the Silhouette Score.
    This method measures how similar an object is to its own cluster compared to other clusters.
    A higher score indicates better-defined clusters.
    """
    silhouette_scores = []
    
    # Iterate through a range of possible cluster numbers
    for k in range(min_clusters, max_clusters + 1):
        # Initialize and fit the K-Means model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        
        # Calculate the silhouette score for the current number of clusters
        score = silhouette_score(data_scaled, kmeans.labels_)
        silhouette_scores.append((score, k))
        
    # Select the number of clusters (k) that resulted in the highest silhouette score
    optimal_k = max(silhouette_scores, key=lambda item: item[0])[1]
    return optimal_k

def create_rfm_segments(rfm_df, n_clusters=None):
    """
    Performs K-Means clustering on RFM data using a robust preprocessing pipeline.
    This involves handling missing values, applying a log transformation to correct
    for skewed data, and then standardizing the data so all features contribute equally.
    
    Args:
        rfm_df (pd.DataFrame): DataFrame containing 'Recency', 'Frequency', and 'Monetary'.
        n_clusters (int, optional): The number of clusters to form. If None,
                                     the function finds an optimal number automatically.
    
    Returns:
        tuple: (
            rfm_df_with_cluster (pd.DataFrame): The original RFM data with a 'Cluster' column.
            cluster_centers_original_scale (pd.DataFrame): The cluster centers transformed back to their original, interpretable scale.
            n_clusters_used (int): The number of clusters used for the final model.
        )
    """
    # Create a new DataFrame for preprocessing to avoid modifying the original
    rfm_processed = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

    # --- Step 1: Handle Missing Values (Imputation) ---
    # This is a crucial first step. We fill any potential missing values (NaNs)
    # with the median of their respective columns. The median is robust to outliers,
    # which are common in RFM data.
    imputer = SimpleImputer(strategy='median')
    rfm_imputed = imputer.fit_transform(rfm_processed)
    rfm_processed = pd.DataFrame(rfm_imputed, 
                                 columns=['Recency', 'Frequency', 'Monetary'], 
                                 index=rfm_df.index)

    # --- Step 2: Logarithmic Transformation ---
    # RFM data is often highly skewed (e.g., many customers buy infrequently, a few buy a lot).
    # A log transform helps to normalize this distribution, making it more suitable for
    # clustering algorithms like K-Means. We add 1 to handle zero values gracefully (log(0) is undefined).
    rfm_log_transformed = pd.DataFrame(index=rfm_processed.index)
    rfm_log_transformed['Recency'] = np.log1p(rfm_processed['Recency'])
    rfm_log_transformed['Frequency'] = np.log1p(rfm_processed['Frequency'])
    rfm_log_transformed['Monetary'] = np.log1p(rfm_processed['Monetary'])

    # --- Step 3: Standardization ---
    # This is the most critical step to address your concern. We scale the log-transformed
    # data to have a mean of 0 and a standard deviation of 1. This ensures that Recency,
    # Frequency, and Monetary all have an equal impact on the clustering algorithm.
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log_transformed)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, 
                                 columns=['Recency', 'Frequency', 'Monetary'], 
                                 index=rfm_df.index)

    # --- Step 4: K-Means Clustering ---
    # Now, we perform K-Means clustering on the fully preprocessed (imputed, logged, and scaled) data.
    if n_clusters is None:
        n_clusters_used = find_optimal_clusters(rfm_scaled_df)
    else:
        n_clusters_used = n_clusters
        
    kmeans = KMeans(n_clusters=n_clusters_used, random_state=42, n_init=10)
    # Assign the resulting cluster label back to the original rfm_df
    rfm_df_with_cluster = rfm_df.copy()
    rfm_df_with_cluster['Cluster'] = kmeans.fit_predict(rfm_scaled_df)
    
    # --- Step 5: Inverse Transformation for Interpretation ---
    # The cluster centers from KMeans are in the scaled, logarithmic space, which is not
    # human-readable. We must perform the inverse transformations to understand them.
    
    # 5a. Inverse the standardization
    cluster_centers_log_scale = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 5b. Inverse the log transformation (using np.expm1, which is exp(x) - 1)
    cluster_centers_original_scale = np.expm1(cluster_centers_log_scale)
    
    # Create a clean DataFrame for the cluster centers
    cluster_centers_df = pd.DataFrame(cluster_centers_original_scale,
                                      columns=['Recency', 'Frequency', 'Monetary'])
    cluster_centers_df['Cluster'] = cluster_centers_df.index
    cluster_centers_df = cluster_centers_df[['Cluster', 'Recency', 'Frequency', 'Monetary']].sort_values(by=['Recency'])

    return rfm_df_with_cluster, cluster_centers_df, n_clusters_used

def get_llm_cluster_names(cluster_centers_json):
    """
    Calls an LLM to assign names and descriptions to clusters.
    """
    try:
        # Define the system prompt
        system_prompt = "You are a world-class marketing analyst specializing in customer segmentation. Your task is to analyze RFM clusters and provide clear, actionable names and descriptions for each segment. Base your analysis solely on the provided JSON data. Do not make any assumptions outside of the data. Your primary goal is to provide accurate and insightful names and descriptions based on the Recency, Frequency, and Monetary values."

        # Define the user prompt
        user_query = f"Here is the average Recency, Frequency, and Monetary value for each of the customer segments:\n\n{cluster_centers_json}\n\n" \
                     "Please provide a descriptive name, a brief description of the customer type, and a marketing recommendation for each segment. " \
                     "The names should be based on common RFM segments, such as 'Champions', 'Loyal Customers', 'About to Sleep', 'At-Risk', 'New Customers', 'Lost Customers', etc. " \
                     "Return a JSON array of objects with the keys 'cluster_id', 'segment_name', 'description', and 'recommendations'."
        
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.7,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and result['choices']:
            json_content = result['choices'][0]['message']['content']
            return json.loads(json_content)
        

    except requests.exceptions.RequestException as e:
        st.error(f"Network or API request error: {e}")
        return []
    except json.JSONDecodeError:
        st.error("LLM response was not a valid JSON object.")
        return {}
