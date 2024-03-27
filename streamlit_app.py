import streamlit as st

st.write('Welcome to Streamlit')

import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Example assuming 3 competitors and 10 variables for simplicity
num_competitors = 3
num_variables = 10

# Create empty DataFrame to hold user input
data = pd.DataFrame(columns=[f'Variable_{i+1}' for i in range(num_variables)])

# Input fields for competitor names
competitor_names = [st.text_input(f'Enter name for Competitor {i+1}: ', key=f'comp_{i}') for i in range(num_competitors)]

# Input fields for scores of each variable for each competitor
for i, name in enumerate(competitor_names):
    if name:  # Proceed only if a name has been entered
        scores = [st.number_input(f'Enter score for {name} - Variable {j+1}: ', min_value=0.0, max_value=1.0, value=0.5, key=f'{i}_{j}') for j in range(num_variables)]
        data.loc[name] = scores

# Button to perform k-means clustering
if st.button('Perform K-Means Clustering'):
    if not data.empty:
        # Assuming the number of clusters is known or determined by another method (e.g., silhouette analysis)
        num_clusters = 2  # Example fixed number of clusters
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(data)

        # Display the clustering result
        for i, name in enumerate(data.index):
            st.write(f'{name} is in Cluster {cluster_labels[i]+1}')
    else:
        st.write("Please enter competitor names and scores to perform clustering.")