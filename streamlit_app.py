import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch  # For defensive barriers
import requests
import json

st.markdown("""
    <style>
    /* Target the text area widgets */
    .stTextArea>div>div>textarea {
        opacity: 1 !important; /* Make text area fully opaque */
        cursor: text !important; /* Change cursor to text selection */
    }
    /* Optional: Adjust the hover state as well */
    .stTextArea>div>div>textarea:hover {
        background-color: #f2f2f2; /* Light grey background on hover, adjust as needed */
    }
    </style>
""", unsafe_allow_html=True)

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

 # Your OpenAI API key
OPENAI_API_KEY = st.secrets["secret"]

# Function to call the OpenAI API
def query_openai_api(data):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}',
    }
    json_data = {
        'model': 'gpt-4-0125-preview',  # Use the correct model you have access to
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data['prompt']}
        ],
    }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)
    
    if response.status_code == 200:
        response_json = response.json()
        try:
            # Directly access the message content in the first choice
            assistant_message_content = response_json['choices'][0]['message']['content']
            return assistant_message_content
        except KeyError as e:
            # Log the error and the unexpected response structure
            st.error(f"KeyError: {e}. Unexpected response structure: {response_json}")
            return "Error: Received unexpected response structure."
    else:
        st.error(f"API request failed with status code {response.status_code}: {response.text}")
        return f"Error: API request failed with status code {response.status_code}"



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
        <p><b>PROACT: Solution Development Atlas: K-Means Competitive Clustering Analysis</b></p>
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
        Reflect on the market share distribution, considering both current market dynamics and potential shifts. Market share can offer insights into a competitor's strength and customer reach. Additionally, for emerging startups and potential entrants not currently in the industry, estimate their market share based on their growth trajectory and anticipated impact over the next 2 to 3 years. This forward-looking approach will help you better prepare for and respond to the evolving competitive landscape.
    """)
    for competitor in st.session_state.competitors:
        if competitor:
            market_share = st.number_input(f'Enter market share for {competitor} (%): ', min_value=0.0, max_value=100.0, value=0.0, step=0.01, format="%.2f", key=f'market_share_{competitor}')
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
    with st.expander("i - Scoring Methodology Guide"):
        st.write("""Evaluate each competitor on pivotal variables that can change the competitive landscape—factors like innovation, customer engagement, and market expansion potential are just the beginning. This evaluation is more than just a scoring exercise; it's a strategic exploration into what makes a company stand out, compete, or even dominate.

**Game-Changers, Creating Advantage, Being Challenged, and Qualifiers:**

**Game-Changers:** These are variables that are not only highly differentiated today but will continue to be in the future. They have the power to create unparalleled competitive advantages and can fundamentally redefine the industry landscape. Pinpoint these factors to ensure your strategy is not just competitive, but dominant.

**Creating Advantage:** Variables that may not be highly differentiated today but are poised to become so in the future fall into this category. These are your hidden gems, opportunities to invest in now for significant payoff later. Identifying and nurturing these aspects can set you on a path to emerging as a leader in uncharted territories.

**Being Challenged:** Highly differentiated today but at risk of becoming less so tomorrow, these variables signal areas where your current strengths might erode. Recognizing them early is crucial for strategic adaptability, allowing you to pivot or bolster these areas to maintain your competitive edge.

**Qualifiers:** Essential for just staying in the game, these variables represent baseline expectations in the industry—lowly differentiated today and tomorrow. While they might not be your leading strategic focus, neglecting them can put you at a disadvantage. Ensure you meet these industry standards even as you strive for distinction elsewhere.

**Balancing Internal and External Factors:** Consider variables from two lenses—internal capabilities such as operational efficiency, technology innovation, and team expertise, and external forces like market trends, customer needs, and regulatory changes. How these factors intersect will guide your strategic focus.

**Weighting for Impact:** Not all variables carry equal weight in determining success. Assign weightings that reflect their potential to drive your competitive advantage. Think critically about how shifts in these variables could impact your positioning and strategy.""")

    # Assuming the number of variables has already been set and is consistent
    num_variables = st.number_input('Enter the number of variables:', min_value=1, value=len(st.session_state.variables) if 'variables' in st.session_state and st.session_state.variables else 3, step=1, key='num_variables_setup')

    # Adjusted to use index in the key for uniqueness
    variables = [st.text_input(f'Enter name for Variable {i+1}: ', value=st.session_state.variables[i] if i < len(st.session_state.variables) else '', key=f'var_name_{i}') for i in range(num_variables)]
    st.session_state.variables = variables

    # Adding helper text about weighting variables
    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Weighting Variables:</strong> Assign weights to each variable reflecting their importance in your analysis. The total weight across all variables should <strong>add up to 100%</strong>. This ensures a balanced approach, where the sum of all weights accurately represents their collective impact on your strategic analysis. Distribute the weights carefully to mirror the significance of each variable in shaping your competitive landscape.</p>
    </div> """, unsafe_allow_html=True)

    # Using index in the key for uniqueness
    variable_weights = {}
    for i, variable in enumerate(variables):
        if variable:  # Ensure the variable has been named
            weight_key = f'weight_{i}_{variable}'  # Incorporate both index and variable name for uniqueness
            weight = st.number_input(f'Weight for {variable} (%):', min_value=0, max_value=100, value=10, key=weight_key)
            variable_weights[variable] = weight
    st.session_state.variable_weights = variable_weights

    st.markdown("""
    <div style="text-align: justify;">
        <p><strong>Scoring Variables:</strong> As you assess each variable, consider the scale carefully. A score of <strong>0</strong> indicates a <strong>major weakness</strong> for an internal variable, or <strong>poor positioning</strong> for an external variable. Conversely, a score of <strong>1</strong> signifies a <strong>major strength</strong> or <strong>superior positioning</strong>. Reflect on each variable's current state and potential to impact your competitive stance, and assign scores accordingly.</p>
    </div>
""", unsafe_allow_html=True)

# Initialize a dictionary in the session state to store scores if it doesn't already exist
    if 'scores' not in st.session_state:
        st.session_state.scores = {}
    
    # Iterate through competitors and variables to display sliders
    for competitor in st.session_state.competitors:
        for variable in st.session_state.variables:
            # Generate a unique key for each score
            score_key = f'score_{competitor}_{variable}'
            # Retrieve the current score from the session state, defaulting to 0.5 if it doesn't exist
            current_score = st.session_state.scores.get(score_key, 0.5)
            
            # Create a slider for the score
            new_score = st.slider(f'Rate {competitor} for {variable}:', min_value=0.0, max_value=1.0, value=current_score, key=score_key)
            
            # Update the score in the session state
            st.session_state.scores[score_key] = new_score

    all_scores_entered = True

    for competitor in st.session_state.competitors:
        for variable in st.session_state.variables:
            score_key = f'score_{competitor}_{variable}'
            # Check if the score_key exists in the session state and if not, set all_scores_entered to False
            if score_key not in st.session_state.scores or st.session_state.scores[score_key] is None:
                all_scores_entered = False

            # Assume current_score is retrieved from session_state as before
            # Assume new_score is set by slider and updated in session_state as before

        # Check if the button is pressed and all scores are entered before proceeding
        if st.button('Score and Analyze') and all_scores_entered:
            # Now you can safely proceed with reconstructing scores_df from session state
            # and your analysis, knowing all scores have been entered
            scores_df = pd.DataFrame(index=st.session_state.competitors, columns=st.session_state.variables)
            for competitor in st.session_state.competitors:
                for variable in st.session_state.variables:
                    score_key = f'score_{competitor}_{variable}'
                    scores_df.at[competitor, variable] = st.session_state.scores[score_key]

            # Proceed with your normalization and analysis as before
            # Remember to set pca_ready and other relevant flags here
    # When ready to analyze, you can reconstruct scores_df from the session state
    if st.button('Score and Analyze') and all_scores_entered:
        # Initialize an empty DataFrame
        scores_df = pd.DataFrame(index=st.session_state.competitors, columns=st.session_state.variables)
        # Populate the DataFrame with scores from the session state
        for competitor in st.session_state.competitors:
            for variable in st.session_state.variables:
                score_key = f'score_{competitor}_{variable}'
                scores_df.at[competitor, variable] = st.session_state.scores.get(score_key, 0.5)

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

    if all_scores_entered:
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
        st.session_state.pca_ready = True  # Set pca_ready flag here
        st.session_state.current_screen = 'show_results'

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
        
        cluster_competitors = {i: [] for i in range(np.unique(st.session_state.cluster_labels).size)}
        for i, competitor in enumerate(st.session_state.competitors):
            cluster_label = st.session_state.cluster_labels[i]
            cluster_competitors[cluster_label].append(competitor)
        
        st.write("Competitors by Cluster:")
        for cluster, competitors in cluster_competitors.items():
            st.write(f"**Cluster {cluster + 1}:** {', '.join(competitors)}")
        
        scaled_data = np.array(st.session_state.scaled_data)
        cluster_labels = np.array(st.session_state.cluster_labels)
        market_share = st.session_state.market_share

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()

        # Plot each competitor
        for i, competitor in enumerate(st.session_state.competitors):
            size = st.session_state.market_share[competitor] * 100
            ax.scatter(principal_components[i, 0], principal_components[i, 1], s=size, label=competitor)
            ax.text(principal_components[i, 0], principal_components[i, 1], competitor, ha='right', va='bottom')

        # Calculate cluster centers
        cluster_centers = np.array([principal_components[st.session_state.cluster_labels == i].mean(axis=0) for i in np.unique(st.session_state.cluster_labels)])
        
        # Determine a dynamic radius for circles based on data spread within each cluster
        for i, center in enumerate(cluster_centers):
            cluster_points = principal_components[st.session_state.cluster_labels == i]
            radius = np.sqrt(((cluster_points - center) ** 2).sum(axis=1).mean())  # RMS distance to center as radius
            ax.add_patch(Circle(center, radius, color='red', fill=False, linestyle='--'))

        # Add offensive arrows showing potential strategic directions
        global_mean = principal_components.mean(axis=0)
        for center in cluster_centers:
            direction = global_mean - center
            ax.annotate('', xy=center + direction * 0.5, xytext=center, arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8))

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Strategic Foresight PROACT Solution Development Atlas')

        st.pyplot(fig)

        pca_contributions = pd.DataFrame(pca.components_, columns=st.session_state.variables, index=['PC1', 'PC2'])
        st.write("PCA Components' Contributions to Variables:")
        st.dataframe(pca_contributions.style.format("{:.2f}"))

        # Assuming 'scores_df' has the competitors as rows and variables as columns with their scores
        scores_df_display = pd.DataFrame(st.session_state['scores_df'], index=st.session_state.competitors, columns=st.session_state.variables)

        # Adding a fallback label column to scores_df_display for each competitor
        fallback_labels = [f'Competitor {i}' for i in range(len(scores_df_display))]
        scores_df_display['Fallback Label'] = fallback_labels

        # You might want to display the 'Fallback Label' column first
        # So let's rearrange the columns to make 'Fallback Label' appear first
        cols = ['Fallback Label'] + [col for col in scores_df_display if col != 'Fallback Label']
        scores_df_display = scores_df_display[cols]

        # Apply formatting only to numerical columns and exclude the 'Fallback Label'
        # First, let's create a style function that applies formatting conditionally
        def apply_style(df):
            return df.style.format("{:.2f}", na_rep="N/A", subset=pd.IndexSlice[:, df.columns.difference(['Fallback Label'])])

        # Use the apply_style function on the DataFrame before displaying it
        st.write("Competitors' Scores for Each Variable:")
        st.dataframe(apply_style(scores_df_display))

        # Now, when getting pca_scores, ensure the 'Fallback Label' column is dropped
        pca_scores = scores_df_display.drop('Fallback Label', axis=1).apply(lambda row: row.to_dict(), axis=1).to_dict()
        
        # Ensure necessary session state variables are initialized
        if 'follow_up_count' not in st.session_state:
            st.session_state.follow_up_count = 0
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Check if all conditions for running the analysis are met (Example: Check if PCA analysis is ready)
        if 'pca_ready' in st.session_state and st.session_state.pca_ready and st.session_state.follow_up_count == 0:
            # Construct the initial prompt for the API with detailed competitor information
            prompt_text = """I have conducted a Principal Component Analysis (PCA) and applied k-means clustering on a dataset representing the competitive landscape in our industry, focusing on various strategic metrics. This combined analysis provides PCA scores for each competitor across critical variables, illustrating their positioning along the principal components PC1 and PC2. It also segments competitors into clusters, offering insights into collective strategic stances within the market. Furthermore, we have visualized this analysis through a strategic map that features defensive barriers around clusters and offensive arrows indicating potential strategic directions.
Given this context, please provide a structured strategic analysis that explores the implications of individual PCA scores, the collective dynamics revealed by k-means clustering, and the strategic insights offered by the defensive barriers and offensive arrows.
Please structure your analysis as follows and in the following order:
1) Key Findings: Begin your analysis with key findings, focusing specifically on the strategic implications of the defensive barriers and offensive arrows as visualized on our strategic map. Please identify which clusters are encircled by defensive barriers and describe what these barriers signify in terms of market defense strategies and competitor cohesion. Similarly, detail the directions indicated by offensive arrows and explicitly name the strategic opportunities or market areas they point towards. This analysis should not only tie back to the PCA and clustering analysis but also provide specific examples of how these visual markers guide our understanding of the competitive landscape.
2) Competitive Analysis: Delve into the combined insights from PCA scores and k-means clustering for each competitor and cluster. For individual competitors, highlight the strategic implications of their scores on PC1 and PC2. For clusters, discuss the common strategic themes or market positions that emerge, and how these groupings reflect broader competitive dynamics. Please refer to competitors by their actual names and not by numbers.It is important that your reply includes competitors names and not competitor 0, competitor 1, etc. 
3) Strategic Recommendations: Conclude with strategic considerations and recommendations informed by the PCA scores, clustering results, and strategic map analysis. Offer insights into potential strategic moves, areas for innovation or differentiation, and considerations for positioning against clusters of competitors.
4) About Methodology: Begin by explaining what the numbers in the PC1 and PC2 components in addition to the competitive scores mean. Provide overview of how PCA scores, particularly in relation to the dimensions PC1 and PC2, can suggest individual strategic positioning. Then, elaborate on how k-means clustering builds upon this by grouping competitors with similar strategic profiles, offering a view of collective competitive dynamics. Discuss the strategic significance of high, low, and negative PCA scores and the insights gained from clustering.
Defensive barriers indicate the spread and cohesion within clusters, showing how competitors collectively defend their strategic positions. Offensive arrows suggest directions for strategic advancement or areas where competitors could potentially disrupt the current competitive equilibrium.
Please incorporate the PCA scores and k-means clustering results for each competitor and cluster into your analysis, ensuring a comprehensive understanding of both individual and collective competitive strategies."""
            for competitor_name, scores in st.session_state['pca_scores'].items():
                prompt_text += f"\nCompetitor '{competitor_name}':\n"
                for variable, score in scores.items():
                    prompt_text += f"- {variable}: {score}\n"
            prompt_text += "\nStart answer going right into the key findings, as if you were briefing a senior executive on the company's most pivotal business decisions."

            # Automatically perform the analysis without waiting for a button click
            api_response_text = query_openai_api({'prompt': prompt_text})
            if not api_response_text.startswith("Error:"):
                st.session_state.conversation_history.append({'prompt': "Initial Analysis", 'response': api_response_text})
                st.session_state.follow_up_count += 1
                # Display the initial analysis response
                st.markdown(f"<div style='margin-bottom: 10px;'><b>Q:</b> Initial Analysis</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px;'><b>A:</b> {api_response_text}</div>", unsafe_allow_html=True)
            else:
                st.error(api_response_text)

        # Display the conversation history using HTML
        for conv in st.session_state.conversation_history:
            st.markdown(f"<div style='margin-bottom: 10px;'><b>Q:</b> {conv['prompt']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px;'><b>A:</b> {conv['response']}</div>", unsafe_allow_html=True)

        # Follow-up question logic
        if st.session_state.follow_up_count == 1:
            follow_up_question = st.text_input("Have a follow-up question? Ask here:")
            if st.button('Ask Follow-Up Question'):
                new_prompt = "\n".join([f"Q: {conv['prompt']}\nA: {conv['response']}" for conv in st.session_state.conversation_history]) + f"\nQ: {follow_up_question}"
                follow_up_response = query_openai_api({'prompt': new_prompt})
                if not follow_up_response.startswith("Error:"):
                    st.session_state.conversation_history.append({'prompt': follow_up_question, 'response': follow_up_response})
                    st.session_state.follow_up_count += 1
                else:
                    st.error(follow_up_response)
        elif st.session_state.follow_up_count > 1:
            # Display a message for further contact after one follow-up question
            st.markdown("""
            Thank you for your engagement! For more questions or to continue the conversation,
            please [contact us](mailto:solutions@strategicforesight.ai).
            """, unsafe_allow_html=True)

  # Subscription Call-to-Action
    st.markdown("### Download White Paper")
    st.markdown("10X Market Impact in 10 Hours with World-Class Foresight")
    
    # Provide the URL to the subscription page
    subscription_url = "https://mailchi.mp/strategicforesight/growth-solutions"
    st.markdown(f"[Download Now]({subscription_url})", unsafe_allow_html=True)

    # Optionally, offer more context or a teaser of what they'll learn
    st.markdown("This white paper will equip you with our 4 move advantage to:")
    st.markdown("""
    - Outmaneuever competitors
    - Outperform markets
    - Outserve customers
    - ...and much more!
    """)

    # Using HTML to center the logo image and add space before the text
    st.markdown("""
    <div style="text-align: center;">
        <img src="https://github.com/strategicfuture/hello-streamlit/blob/main/10%20Hour%20Blueprint%20(1).png?raw=true" width="600" style="margin-bottom: 20px;">
    </div> """, unsafe_allow_html=True)

    # Adding helper text about weighting variables
    st.markdown("""
    <div style="text-align: justify;">
        <p><strong><b><h3>Solution Development Atlas Reflection Questions:</h3></b></strong> 
                
<b>Strategic Positioning:</b> How does your company's position on the map reflect your current competitive advantage? Are there variables where you lead, and how sustainable are these advantages?

<b>Emerging Opportunities:</b> Based on the plot, which emerging opportunities can your company uniquely capitalize on? How can you leverage these to redefine or expand your market presence?

<b>Future Differentiation:</b> Which variables are likely to become more critical for differentiation in the future? How can your company proactively develop capabilities or solutions in these areas?

<b>Adaptability to Change:</b> Considering potential industry shifts highlighted by the map, how adaptable is your current strategy? What changes might you need to consider to remain competitive or become a leader?

<b>Innovation and Disruption:</b> Are there areas ripe for innovation or disruption where you can challenge established competitors or create new value for customers?

<b>Resource Allocation:</b> How should your findings from the map influence the allocation of your resources (time, talent, capital) towards areas of high strategic importance?

<b>Collaboration and Partnership:</b> Could collaborations or partnerships enhance your positioning on certain variables or accelerate your path to leadership?

<b>Potential Threats:</b> Which companies positioned closely to you pose the most significant threats, and why? How can you monitor these competitors and prepare for possible strategic moves they might make?

<b>Blind Spots:</b> Were there any surprises or “blind spots” revealed by the strategic group map? How can your company address these gaps in perception or strategy?

<b>Long-term Vision:</b> How does your positioning align with your company's long-term vision and goals? What strategic shifts might you need to consider to ensure alignment and success in the future?</p>
    </div> """, unsafe_allow_html=True)

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


