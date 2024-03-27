import streamlit as st

st.write('Welcome to Streamlit')

import streamlit as st
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to find the optimal number of clusters using the elbow method
def find_optimal_clusters(data):
    inertias = []
    K_range = range(1, min(len(data), 11))  # Assuming a max of 10 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    if len(inertias) > 1:
        deltas = np.diff(inertias)
        elbow_point = np.argmin(deltas) + 2  # +2 to adjust for the shift caused by np.diff and 0-indexing
    else:
        elbow_point = 1

    return elbow_point

# Function to display the first screen for entering competitor and variable names
def enter_info():
    with st.form("competitor_variables"):
        num_competitors = st.number_input('Enter the number of competitors:', min_value=1, value=3, step=1)
        competitor_names = [st.text_input(f'Enter name for Competitor {i+1}: ', key=f'comp_{i}') for i in range(num_competitors)]
        variables = [st.text_input(f'Enter name for Variable {i+1}: ', key=f'var_{i}') for i in range(10)]
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.competitors = competitor_names
        st.session_state.variables = variables
        st.session_state.current_screen = 'score_variables'

# Function to display the second screen for scoring variables
def score_variables():
    data = pd.DataFrame(columns=st.session_state.variables)
    with st.form("scoring"):
        for competitor in st.session_state.competitors:
            if competitor:  # Proceed only if a name has been entered
                scores = [st.slider(f'Enter score for {competitor} - {variable}: ', min_value=0.0, max_value=1.0, value=0.5, key=f'{competitor}_{variable}') for variable in st.session_state.variables]
                data.loc[competitor] = scores
        submitted = st.form_submit_button("Analyze")

    if submitted and not data.empty:
        optimal_clusters = find_optimal_clusters(data)
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Display the clustering result
        st.write(f'Optimal number of clusters determined by elbow method: {optimal_clusters}')
        for i, competitor in enumerate(data.index):
            st.write(f'{competitor} is in Cluster {cluster_labels[i]+1}')

# Initialize session state for screen navigation
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 'enter_info'

# Display screens based on the current state
if st.session_state.current_screen == 'enter_info':
    enter_info()
elif st.session_state.current_screen == 'score_variables':
    score_variables()
