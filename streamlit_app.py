import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def find_optimal_clusters(data):
    inertias = []
    K_range = range(1, min(len(data), 11) + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    deltas = np.diff(inertias)
    if len(deltas) > 0:
        elbow_point = np.argmin(deltas) + 2
    else:
        elbow_point = 1
    return elbow_point

def main():
    st.title("Dynamic K-Means Clustering Visualization")

    with st.form("data_input"):
        num_competitors = st.number_input('Enter the number of competitors:', min_value=2, max_value=10, value=3, step=1)
        variable_names = [f'Variable {i+1}' for i in range(10)]  # Example variable names
        data = {}

        for i in range(int(num_competitors)):
            competitor_name = st.text_input(f'Competitor {i+1} Name:', key=f'comp_{i}')
            scores = st.text_input(f'Enter scores for {competitor_name} (comma-separated):', key=f'scores_{i}')
            data[competitor_name] = [float(score) for score in scores.split(',') if score]

        submitted = st.form_submit_button("Submit")

    if submitted and data:
        df = pd.DataFrame(data, index=variable_names).T
        optimal_clusters = find_optimal_clusters(df)
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(df)

        st.write(f'Optimal number of clusters determined by elbow method: {optimal_clusters}')

        plot_choice = st.radio("How would you like to choose axes for plotting?",
                               ('Use PCA to determine axes automatically', 'Manually select variables for axes'))

        if plot_choice == 'Use PCA to determine axes automatically':
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(df)
            fig, ax = plt.subplots()
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
            st.pyplot(fig)

        elif plot_choice == 'Manually select variables for axes':
            x_var = st.selectbox('Select variable for X-axis:', options=variable_names)
            y_var = st.selectbox('Select variable for Y-axis:', options=variable_names, index=1 if len(variable_names) > 1 else 0)
            fig, ax = plt.subplots()
            scatter = ax.scatter(df[x_var], df[y_var], c=labels, cmap='viridis')
            st.pyplot(fig)

if __name__ == '__main__':
    main()
