import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Initialize session state variables if not already present
if 'competitors' not in st.session_state:
    st.session_state.competitors = []
if 'variables' not in st.session_state:
    st.session_state.variables = []
if 'market_share' not in st.session_state:
    st.session_state.market_share = {}  # Initialize market share dictionary
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 'enter_info'
if 'show_plot' not in st.session_state:
    st.session_state.show_plot = False
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None

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

# Function to display the first screen for entering competitor names
def enter_info():
    num_competitors = st.number_input('Enter the number of competitors:', min_value=1, value=3, step=1, key='num_competitors')
    competitor_names = [st.text_input(f'Enter name for Competitor {i+1}: ', key=f'comp_{i}') for i in range(num_competitors)]
    st.session_state.competitors = competitor_names
    if st.button('Next to Enter Market Share'):
        st.session_state.current_screen = 'enter_market_share'

# Function to enter market share percentages for each competitor
def enter_market_share():
    if st.button('Back to Enter Info'):
        st.session_state.current_screen = 'enter_info'
    for competitor in st.session_state.competitors:
        if competitor:  # Only proceed if a name has been entered
            market_share = st.number_input(f'Enter market share for {competitor} (%): ', min_value=0, max_value=100, key=f'market_share_{competitor}')
            st.session_state.market_share[competitor] = market_share
    if st.button('Next to Score Variables'):
        st.session_state.current_screen = 'score_variables'

# Function to display the second screen for scoring variables
def score_variables():
    if st.button('Back to Enter Market Share'):
        st.session_state.current_screen = 'enter_market_share'
    num_variables = st.number_input('Enter the number of variables:', min_value=1, value=10, step=1, key='num_variables')
    variables = [st.text_input(f'Enter name for Variable {i+1}: ', key=f'var_{i}') for i in range(num_variables)]
    st.session_state.variables = variables
    
    # Check if variables have been named before showing sliders
    if len(variables) == num_variables and all(variables):  # Ensure all variable names are entered
        scores_df = pd.DataFrame(columns=variables)
        for competitor in st.session_state.competitors:
            if competitor:  # Only proceed if a name has been entered
                scores = [st.slider(f'Enter score for {competitor} - {variable}: ', min_value=0.0, max_value=1.0, value=0.5, key=f'{competitor}_{variable}') for variable in variables]
                scores_df.loc[competitor] = scores
        # "Score and Analyze" button to trigger analysis and move to results
        if st.button('Score and Analyze'):
            # Perform scaling and clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(scores_df)
            optimal_clusters = find_optimal_clusters(scaled_data)
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)

            # Save results for use in show_results
            st.session_state.scaled_data = scaled_data
            st.session_state.cluster_labels = cluster_labels
            st.session_state.show_plot = True
            st.session_state['scores_df'] = scores_df.to_dict('list')  # Convert DataFrame to a dictionary for session state storage
            st.session_state.current_screen = 'show_results'
    else:
        st.info("Please enter names for all variables before scoring.")


# Function to display the results and plotting
def show_results():
    if st.button('Back to Score Variables'):
        st.session_state.current_screen = 'score_variables'
    
    if st.session_state.show_plot:
        scores_df = pd.DataFrame(st.session_state['scores_df']) 
        scores_df.columns = st.session_state.variables  # Ensure columns match selected variables
        
        plot_choice = st.radio("How would you like to choose axes for plotting?",
                               ('Use PCA to determine axes automatically', 'Manually select variables for axes'))
        
        scaled_data = np.array(st.session_state.scaled_data)
        cluster_labels = np.array(st.session_state.cluster_labels)
        market_share = st.session_state.market_share  # Retrieve market share data
        
        if plot_choice == 'Use PCA to determine axes automatically':
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            fig, ax = plt.subplots()
            for i, competitor in enumerate(st.session_state.competitors):
                ax.scatter(principal_components[i, 0], principal_components[i, 1], s=market_share[competitor] * 100, label=competitor)
                ax.annotate(competitor, (principal_components[i, 0], principal_components[i, 1]))
            st.pyplot(fig)
            
            # Display PCA component contributions
            pca_contributions = pd.DataFrame(pca.components_, columns=st.session_state.variables, index=['PC1', 'PC2'])
            st.write("PCA Components' Contributions to Variables:")
            st.dataframe(pca_contributions.style.format("{:.2f}"))
            
        elif plot_choice == 'Manually select variables for axes':
            variable_options = st.session_state.variables
            x_var = st.selectbox('Select variable for X-axis:', options=variable_options)
            y_var = st.selectbox('Select variable for Y-axis:', options=variable_options, index=1 if len(variable_options) > 1 else 0)
            fig, ax = plt.subplots()
            for competitor in st.session_state.competitors:
                ax.scatter(scores_df[x_var][competitor], scores_df[y_var][competitor], s=market_share[competitor] * 100, label=competitor)
                ax.annotate(competitor, (scores_df[x_var][competitor], scores_df[y_var][competitor]))
            st.pyplot(fig)
    else:
        st.error("Please go back and perform clustering first.")

# App layout based on current screen
if st.session_state.current_screen == 'enter_info':
    enter_info()
elif st.session_state.current_screen == 'enter_market_share':
    enter_market_share()
elif st.session_state.current_screen == 'score_variables':
    score_variables()
elif st.session_state.current_screen == 'show_results':
    show_results()