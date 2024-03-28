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
if 'variable_weights' not in st.session_state:
    st.session_state.variable_weights = {}
if 'market_share' not in st.session_state:
    st.session_state.market_share = {}  # Initialize market share dictionary
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 'init_challenge'
if 'show_plot' not in st.session_state:
    st.session_state.show_plot = False
if 'scaled_data' not in st.session_state:
    st.session_state.scaled_data = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None

# New initial challenge screen function
def init_challenge_screen():
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/Strategic%20Foresight%20Logo%20Suite-02.png?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)
    st.title("Your Four Move Advantage for Sustainable Growth")
    challenge_response = st.radio("## How can addressing your most significant current challenge spark new growth?",
                                  options=[
                                      "PREDICT: Are you ready to enter or expand your reach into market segments where evolving customer needs align with your distinctive capabilities?",
                                      "PROACT: Do you aim to craft standout solutions, distinguishing your competitive edge in product development uncharted white space?",
                                      "PERFORM: With the best fit market segments identified, are you strategizing how to allocate your investments across geos and industries for maximum impact?",
                                      "OUTPACE: Are you prepared to embed scenario planning into your growth strategies, securing a first-mover advantage in future-proofing against unforeseen shifts?"
                                  ], index=1, key='init_challenge')  # Default to the second option

    if st.button("Next"):
        if challenge_response.startswith("PROACT"):
            st.session_state.current_screen = 'enter_info'
        else:
            st.info("This demo is designed to spotlight PROACT strategies for developing standout competitive solutions. For a deeper dive and to continue this enriching conversation, please visit us at strategicforesight.ai")
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/usvsthem.jpg?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)


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

def apply_min_max_normalization(scores_df, variable_weights):
    # Normalize variable weights so they sum to 1
    total_weight = sum(variable_weights.values())
    normalized_weights = {k: v / total_weight for k, v in variable_weights.items()}
    # Apply weighted min-max normalization to each variable
    for variable in scores_df.columns:
        min_val = scores_df[variable].min()
        max_val = scores_df[variable].max()
        range_val = max_val - min_val
        if range_val > 0:  # Avoid division by zero
            scores_df[variable] = scores_df[variable].apply(lambda x: (x - min_val) / range_val) * normalized_weights[variable]
        else:
            scores_df[variable] = 0  # Handle case where all values are the same
    return scores_df

def enter_info():
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/Strategic%20Foresight%20Logo%20Suite-02.png?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)
    if st.button('Return to Start'):
        st.session_state.current_screen = 'init_challenge'
    st.header("Enter Competitors")
    # Helper text
    st.markdown("""
        Consider not just your direct competitors but also emerging startups and established leaders in adjacent spaces. These players could redefine the competitive landscape and present new challenges or opportunities.
    """)
    num_competitors = st.number_input('Enter the number of competitors:', min_value=1, value=3, step=1, key='num_competitors')
    competitor_names = [st.text_input(f'Enter name for Competitor {i+1}: ', key=f'comp_{i}') for i in range(num_competitors)]
    st.session_state.competitors = competitor_names
    if st.button('Next to Enter Market Share'):
        st.session_state.current_screen = 'enter_market_share'

def enter_market_share():
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/Strategic%20Foresight%20Logo%20Suite-02.png?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)
    if st.button('Back to Enter Info'):
        st.session_state.current_screen = 'enter_info'
    st.header("Enter Market Share")
    # Helper text
    st.markdown("""
        Reflect on the market share distribution, considering both current market dynamics and potential shifts. Market share can offer insights into a competitor's strength and customer reach.
    """)
    for competitor in st.session_state.competitors:
        if competitor:
            market_share = st.number_input(f'Enter market share for {competitor} (%): ', min_value=0, max_value=100, key=f'market_share_{competitor}')
            st.session_state.market_share[competitor] = market_share
    if st.button('Next to Score Variables'):
        st.session_state.current_screen = 'score_variables'

def score_variables():
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/Strategic%20Foresight%20Logo%20Suite-02.png?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)
    if st.button('Back to Enter Market Share'):
        st.session_state.current_screen = 'enter_market_share'
    st.header("Score Variables")
    # Helper text
    st.markdown("""
        Evaluate each competitor on pivotal variables that can alter the competitive landscape—factors like innovation, customer engagement, and market expansion potential are just the beginning. This evaluation is more than just a scoring exercise; it's a strategic exploration into what makes a company stand out, compete, or even dominate.

**Game-Changers, Creating Advantage, Being Challenged, and Qualifiers:**

**Game-Changers:** These are variables that are not only highly differentiated today but will continue to be in the future. They have the power to create unparalleled competitive advantages and can fundamentally redefine the industry landscape. Pinpoint these factors to ensure your strategy is not just competitive, but dominant.

**Creating Advantage:** Variables that may not be highly differentiated today but are poised to become so in the future fall into this category. These are your hidden gems, opportunities to invest in now for significant payoff later. Identifying and nurturing these aspects can set you on a path to emerging as a leader in uncharted territories.

**Being Challenged:** Highly differentiated today but at risk of becoming less so tomorrow, these variables signal areas where your current strengths might erode. Recognizing them early is crucial for strategic adaptability, allowing you to pivot or bolster these areas to maintain your competitive edge.

**Qualifiers:** Essential for just staying in the game, these variables represent baseline expectations in the industry—lowly differentiated today and tomorrow. While they might not be your leading strategic focus, neglecting them can put you at a disadvantage. Ensure you meet these industry standards even as you strive for distinction elsewhere.

**Balancing Internal and External Factors:** Consider variables from two lenses—internal capabilities such as operational efficiency, technology innovation, and team expertise, and external forces like market trends, customer needs, and regulatory changes. How these factors intersect will guide your strategic focus.

**Weighting for Impact:** Not all variables carry equal weight in determining success. Assign weightings that reflect their potential to drive your competitive advantage. Think critically about how shifts in these variables could impact your positioning and strategy.
    """)
    # Assuming the number of variables has already been set and is consistent
    num_variables = st.number_input('Enter the number of variables:', min_value=1, value=len(st.session_state.variables) if 'variables' in st.session_state and st.session_state.variables else 3, step=1, key='num_variables_setup')

    # Adjusted to use index in the key for uniqueness
    variables = [st.text_input(f'Enter name for Variable {i+1}: ', value=st.session_state.variables[i] if i < len(st.session_state.variables) else '', key=f'var_name_{i}') for i in range(num_variables)]
    st.session_state.variables = variables

    # Using index in the key for uniqueness
    variable_weights = {}
    for i, variable in enumerate(variables):
        if variable:  # Ensure the variable has been named
            weight_key = f'weight_{i}_{variable}'  # Incorporate both index and variable name for uniqueness
            weight = st.number_input(f'Weight for {variable} (%):', min_value=0, max_value=100, value=10, key=weight_key)
            variable_weights[variable] = weight
    st.session_state.variable_weights = variable_weights

    scores_df = pd.DataFrame(index=st.session_state.competitors, columns=st.session_state.variables)
    all_scores_entered = True
    for competitor in st.session_state.competitors:
        for variable in st.session_state.variables:
            if competitor and variable:
                score_key = f'{competitor}_{variable}'
                score = st.slider(f'Rate {competitor} for {variable}:', 0.0, 1.0, 0.5, key=score_key)
                scores_df.at[competitor, variable] = score
            else:
                all_scores_entered = False

    if all_scores_entered and st.button('Score and Analyze'):
        scores_df = apply_min_max_normalization(scores_df, st.session_state.variable_weights)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(scores_df.fillna(0))
        optimal_clusters = find_optimal_clusters(scaled_data)
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        st.session_state.scaled_data = scaled_data
        st.session_state.cluster_labels = cluster_labels
        st.session_state.show_plot = True
        st.session_state.scores_df = scores_df.to_dict('list')
        st.session_state.current_screen = 'show_results'
    elif not all_scores_entered:
        st.warning('Please enter names for all competitors and variables.')

def show_results():
    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/Strategic%20Foresight%20Logo%20Suite-02.png?raw=true" width="350">
        </div>
    """, unsafe_allow_html=True)
    if st.button('Back to Score Variables'):
        st.session_state.current_screen = 'score_variables'
    
    if st.session_state.show_plot:
        scores_df = pd.DataFrame(st.session_state['scores_df'])
        scores_df.columns = st.session_state.variables
        
        num_clusters = np.unique(st.session_state.cluster_labels).size
        st.write(f"Number of clusters: {num_clusters}")
        for i, competitor in enumerate(st.session_state.competitors):
            st.write(f"{competitor} is in Cluster {st.session_state.cluster_labels[i]+1}")

        plot_choice = st.radio("How would you like to choose axes for plotting?", ('Use PCA to determine axes automatically', 'Manually select variables for axes'))
        
        scaled_data = np.array(st.session_state.scaled_data)
        cluster_labels = np.array(st.session_state.cluster_labels)
        market_share = st.session_state.market_share
        
        if plot_choice == 'Use PCA to determine axes automatically':
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(scaled_data)
            fig, ax = plt.subplots()
            for i, competitor in enumerate(st.session_state.competitors):
                ax.scatter(principal_components[i, 0], principal_components[i, 1], s=market_share[competitor] * 100, label=competitor)
                ax.annotate(competitor, (principal_components[i, 0], principal_components[i, 1]))
            st.pyplot(fig)
            
            pca_contributions = pd.DataFrame(pca.components_, columns=st.session_state.variables, index=['PC1', 'PC2'])
            st.write("PCA Components' Contributions to Variables:")
            st.dataframe(pca_contributions.style.format("{:.2f}"))
            
        elif plot_choice == 'Manually select variables for axes':
            try:
                variable_options = st.session_state.variables
                x_var = st.selectbox('Select variable for X-axis:', options=variable_options)
                y_var = st.selectbox('Select variable for Y-axis:', options=variable_options, index=1 if len(variable_options) > 1 else 0)
                fig, ax = plt.subplots()
                for competitor in st.session_state.competitors:
                    x_score = scores_df.loc[competitor, x_var]
                    y_score = scores_df.loc[competitor, y_var]
                    market_size = market_share[competitor] * 100
                    ax.scatter(x_score, y_score, s=market_size, label=competitor)
                    ax.annotate(competitor, (x_score, y_score))
                st.pyplot(fig)
            except KeyError as e:
                st.error(f"An error occurred due to too much convergence among the selected axes or missing data: {e}. Please reconsider the variables chosen for axes or ensure all competitors and variables have been scored.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}. Please check your data and selections.")
    else:
        st.error("Please go back and perform clustering first.")


  # Subscription Call-to-Action
    st.markdown("### Download White Paper")
    st.markdown("10X Market Impact in 10 Hours with World-Class Foresight")
    
    # Provide the URL to the subscription page
    subscription_url = "https://mailchi.mp/strategicforesight/growth-solutions"
    st.markdown(f"[Download Now]({subscription_url})", unsafe_allow_html=True)

    # Optionally, offer more context or a teaser of what they'll learn
    st.markdown("### This white paper will equip you with our 4 move advantage to:")
    st.markdown("""
    - Outmaneuever competitors
    - Outperform markets
    - Outserve customers
    - ...and much more!
    """)

    # Using HTML to center the logo image
    st.markdown("""
        <div style="text-align: center;">
            <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/10%20Hour%20Blueprint%20(1).png?raw=true" width="600">
        </div>
    """, unsafe_allow_html=True)

# App layout based on current screen
if st.session_state.current_screen == 'init_challenge':
    init_challenge_screen()
elif st.session_state.current_screen == 'enter_info':
    enter_info()
elif st.session_state.current_screen == 'enter_market_share':
    enter_market_share()
elif st.session_state.current_screen == 'score_variables':
    score_variables()
elif st.session_state.current_screen == 'show_results':
    show_results()