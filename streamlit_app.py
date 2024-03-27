import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Initialize session state variables if not already present
if 'competitors' not in st.session_state:
    st.session_state['competitors'] = []
if 'variables' not in st.session_state:
    st.session_state['variables'] = []
if 'current_screen' not in st.session_state:
    st.session_state['current_screen'] = 'enter_info'

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
    num_competitors = st.number_input('Enter the number of competitors:', min_value=1, value=3, step=1, key='num_competitors')
    num_variables = st.number_input('Enter the number of variables:', min_value=1, value=10, step=1, key='num_variables')
    
    # Input fields for competitor names
    competitor_names = [st.text_input(f'Enter name for Competitor {i+1}: ', key=f'comp_{i}') for i in range(int(num_competitors))]
    st.session_state['competitors'] = competitor_names
    
    # Input fields for variable names
    variables = [st.text_input(f'Enter name for Variable {i+1}: ', key=f'var_{i}') for i in range(int(num_variables))]
    st.session_state['variables'] = variables

    if st.button('Next to Score Variables'):
        st.session_state['current_screen'] = 'score_variables'

# Function to display the second screen for scoring variables
def score_variables():
    # Create a DataFrame to hold the scores
    scores_df = pd.DataFrame(columns=st.session_state['variables'])
    
    # Collect scores for each competitor
    for competitor in st.session_state['competitors']:
        if competitor:  # Proceed only if a name has been entered
            scores = [st.slider(f'Enter score for {competitor} - {variable}: ', min_value=0.0, max_value=1.0, value=0.5, key=f'{competitor}_{variable}') for variable in st.session_state['variables']]
            scores_df.loc[competitor] = scores

    if st.button('Perform K-Means Clustering'):
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(scores_df)

        # Use the elbow method to find the optimal number of clusters
        optimal_clusters = find_optimal_clusters(scaled_data)
        st.write(f'Optimal number of clusters determined by elbow method: {optimal_clusters}')
        
        # Perform K-Means Clustering with the determined optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Display the clustering result
        for i, competitor in enumerate(scores_df.index):
            st.write(f'{competitor} is in Cluster {cluster_labels[i]+1}')

            plot_choice = st.radio("How would you like to choose axes for plotting?",
                               ('Use PCA to determine axes automatically', 'Manually select variables for axes'))

        if plot_choice == 'Use PCA to determine axes automatically':
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            fig, ax = plt.subplots()
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=cluster_labels, cmap='viridis')

            # Optional: annotate points with competitor names
            for i, competitor in enumerate(scores_df.index):
                ax.annotate(competitor, (principal_components[i, 0], principal_components[i, 1]))

            st.pyplot(fig)

        elif plot_choice == 'Manually select variables for axes':
            variable_options = st.session_state['variables']  # Fetching variable names from session state
            x_var = st.selectbox('Select variable for X-axis:', options=variable_options)
            y_var = st.selectbox('Select variable for Y-axis:', options=variable_options, index=1 if len(variable_options) > 1 else 0)
            fig, ax = plt.subplots()
            scatter = ax.scatter(scores_df[x_var], scores_df[y_var], c=cluster_labels, cmap='viridis')

            # Optional: annotate points with competitor names
            for i, competitor in enumerate(scores_df.index):
                ax.annotate(competitor, (scores_df.at[competitor, x_var], scores_df.at[competitor, y_var]))

            st.pyplot(fig)

# App layout based on current screen
if st.session_state['current_screen'] == 'enter_info':
    enter_info()
elif st.session_state['current_screen'] == 'score_variables':
    score_variables()
