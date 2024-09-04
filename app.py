import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load dataset
def load_data(data):
    df = pd.read_csv(data)
    return df

# Function to vectorize text and compute cosine similarity matrix
def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

@st.cache_data
def get_recommendation(title, cosine_sim_mat, df, rec_type='Most Popular'):
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    idx = course_indices.get(title, None)
    
    if idx is None:
        st.warning("Course title not found in the dataset.")
        return pd.DataFrame()  # Return empty DataFrame if course title not found
    
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    if rec_type == 'Most Popular':
        selected_course_indices = [i[0] for i in sim_scores[1:11]]  # Top 10 similar courses
    elif rec_type == 'Newest':
        selected_course_indices = df.sort_values(by='date_added', ascending=False).head(10).index.tolist()
    elif rec_type == 'Highest Rated':
        selected_course_indices = df.sort_values(by='rating', ascending=False).head(10).index.tolist()
    
    result_df = df.iloc[selected_course_indices].head(10)  # Ensure this line limits the number of courses to 10
    result_df['similarity_score'] = [i[1] for i in sim_scores[1:11]]
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses

# HTML template for displaying results with enhanced styling and icons
RESULT_TEMP = """
<div style="width:100%;height:100%;margin:5px;padding:10px;position:relative;border-radius:10px;
box-shadow:0 0 15px rgba(0, 150, 136, 0.3); background-color: #ffffff; border-left: 5px solid #009688; margin-bottom: 20px;
transition: transform 0.3s ease, box-shadow 0.3s ease;">
<h4 style="color:#009688; margin: 0;">{}</h4>
<p style="color:#333; margin: 5px 0;"><span style="color:#009688;">🔍 Similarity Score:</span> {}</p>
<p style="color:#333; margin: 5px 0;"><span style="color:#009688;">🔗</span> <a href="{}" target="_blank" style="color:#009688;">Course Link</a></p>
<p style="color:#333; margin: 5px 0;"><span style="color:#009688;">💰 Price:</span> {}</p>
<p style="color:#333; margin: 5px 0;"><span style="color:#009688;">👥 Students Enrolled:</span> {}</p>
</div>
"""

# Function to get top-rated courses
@st.cache_data
def get_top_rated_courses(df, num_of_courses=10):
    top_rated_df = df[df['price'] > 0]  # Filter out courses with price 0
    top_rated_df = top_rated_df.sort_values(by='num_subscribers', ascending=False).head(num_of_courses)
    return top_rated_df[['course_title', 'url', 'price', 'num_subscribers']]

# Function to search term if not found
@st.cache_data
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False)]
    return result_df

# Main function for Streamlit app
def main():
    # Set page config at the start
    st.set_page_config(page_title="Course Recommendation App", page_icon="🎓")

    # Inject custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: darkturquoise;
    }
    /* Background image for the whole page */
    .css-1f3v6nr {
        background-image: url('https://img.freepik.com/free-vector/education-technology-futuristic-background-vector-gradient-blue-digital-remix_53876-114092.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    /* Custom styling for sidebar */
    .css-1d391kg {
        background-color: #00796b;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    /* Custom styling for the menu items */
    .css-1n1n7f2 {
        padding: 10px;
        border-radius: 10px;
    }
    /* Custom styling for the content */
    .css-1f3v6nr {
        color: #333;
    }
    .css-1r6slbq {
        color: #0073e6;
    }
    /* Styling for the header */
    .css-1d391kg h1 {
        color: #fff;
    }
    /* Change button and box active color to darkturquoise */
    button:active, .stButton>button:focus {
        background-color: #00796b !important;
        color: white !important;
    }
    .css-1aumxhk:hover {
        background-color: #00796b !important;
        color: white !important;
    }
    .st-expander-header:focus {
        background-color: #00796b !important;
        color: white !important;
    }
    /* Animation for recommendation cards */
    .recommendation-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 150, 136, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    /* Custom hover effect */
    .recommendation-card {
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Course Recommendation App")
    st.markdown("Welcome to the **Ultimate Course Finder**! Discover the perfect courses tailored to your passions and goals.")
    
    # Sidebar Menu
    menu = ["🏠 Home", "🔍 Recommend", "📘 About"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)
    
    # State management for toggling
    if 'show_top_rated' not in st.session_state:
        st.session_state['show_top_rated'] = False
    
    # Top Rated Courses button right below the menu with toggle functionality
    if st.sidebar.button("🎓 Top Rated Courses"):
        st.session_state['show_top_rated'] = not st.session_state['show_top_rated']

    # Load dataset
    df = load_data("data/udemy_course_data.csv")
    
    if choice == "🏠 Home":
        st.subheader("🏠 Home")
        st.markdown(" Explore a curated selection of top courses from our extensive collection. Dive in and start learning today!")
        st.dataframe(df.head(10))
    
    elif choice == "🔍 Recommend":
        st.subheader("🔍 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Enter Course Title: Discover courses that match your interests.")
        rec_type = st.sidebar.selectbox("Select Recommendation Type", ["Most Popular", "Newest", "Highest Rated"])
        
        if st.button("Recommend"):
            if search_term:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, rec_type)
                    if not results.empty:
                        st.markdown("### 🎯 Recommendations")
                        with st.expander("Results as JSON"):
                            results_json = results.to_dict('index')
                            st.json(results_json)
                        
                        for _, row in results.iterrows():
                            rec_title = row['course_title']
                            rec_score = row['similarity_score']
                            rec_url = row['url']
                            rec_price = row['price']
                            rec_num_sub = row['num_subscribers']
                            stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250, class_="recommendation-card")
                    else:
                        st.warning("No recommendations found for the selected type.")
                
                except KeyError:
                    # Search for similar courses only if exact match is not found
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.info("Suggested Options:")
                        st.dataframe(result_df)
                    else:
                        st.warning("Course not found. Please try a different search term.")
    
    elif choice == "📘 About":
        st.subheader("📘 About This App")
        st.markdown("""
        This app uses advanced machine learning algorithms to recommend courses based on your search preferences. 
        We leverage cosine similarity to find courses similar to your input and provide top suggestions based on popularity, 
        newest additions, or highest ratings.
        """)
    
    if st.session_state['show_top_rated']:
        st.subheader("🌟 Top Rated Courses")
        top_rated_courses = get_top_rated_courses(df)
        for _, row in top_rated_courses.iterrows():
            course_title = row['course_title']
            course_url = row['url']
            course_price = row['price']
            course_subscribers = row['num_subscribers']
            st.markdown(f"**{course_title}**\n[Course Link]({course_url})\nPrice: {course_price}\nSubscribers: {course_subscribers}\n")

if __name__ == '__main__':
    main()
