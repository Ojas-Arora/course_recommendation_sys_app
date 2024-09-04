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

# Recommendation system function with caching
@st.cache_data
def get_recommendation(title, cosine_sim_mat, df, num_of_rec=10):
    course_indices = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    idx = course_indices[title]
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = [i[1] for i in sim_scores[1:num_of_rec+1]]
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']].head(num_of_rec)
    return final_recommended_courses

# HTML template for displaying results with enhanced styling and icons
RESULT_TEMP = """
<div style="width:100%;height:100%;margin:5px;padding:10px;position:relative;border-radius:5px;
box-shadow:0 0 10px 2px darkturquoise; background-color: #ffffff; border-left: 5px solid darkturquoise; margin-bottom: 20px;">
<h4 style="color:darkturquoise;">{}</h4>
<p style="color:darkturquoise;"><span style="color:#333;">🔍 Similarity Score:</span> {}</p>
<p style="color:darkturquoise;"><span style="color:#333;">🔗</span> <a href="{}" target="_blank">Course Link</a></p>
<p style="color:darkturquoise;"><span style="color:#333;">💰 Price:</span> {}</p>
<p style="color:darkturquoise;"><span style="color:#333;">👥 Students Enrolled:</span> {}</p>
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
        background-image: url('https://wallpapers.com/images/hd/abstract-blueish-white-color-nrvpjoky2673bptv.jpg');
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
        background-color: #0073e6;
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
        background-color: darkturquoise !important;
        color: white !important;
    }
    .css-1aumxhk:hover {
        background-color: darkturquoise !important;
        color: white !important;
    }
    .st-expander-header:focus {
        background-color: darkturquoise !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Course Recommendation App")
    st.markdown("Welcome to the **Course Recommendation App**! Find courses tailored to your interests.")
    
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
        st.markdown("Browse the first few courses from our dataset:")
        st.dataframe(df.head(10))
    
    elif choice == "🔍 Recommend":
        st.subheader("🔍 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Search for a course by title")
        num_of_rec = st.sidebar.slider("Number of Recommendations", 4, 30, 7)
        
        if st.button("Recommend"):
            if search_term:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
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
                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)
                
                except KeyError:
                    # Search for similar courses only if exact match is not found
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.info("Suggested Options:")
                        st.dataframe(result_df)
                    else:
                        st.warning("Course not found. Please try a different search term.")
    
    elif choice == "📘 About":
        st.subheader("📘 About")
        st.markdown("""
        This **Course Recommendation App** helps you discover the best courses tailored to your needs and interests.
        
        - **🔍 Search for Courses:** Use the search functionality to find courses similar to what you are interested in.
        - **🎓 Top Rated Courses:** Check out the highest-rated courses based on user reviews and ratings.
        
        Built using **Streamlit** and **Pandas**, this app is a demonstration of a basic recommendation system.
        """, unsafe_allow_html=True)
    
    # Display Top Rated Courses if button is clicked and toggled on
    if st.session_state['show_top_rated']:
        st.subheader("🎓 Top Rated Courses")
        top_courses = get_top_rated_courses(df)
        st.markdown("### 📊 Top Courses Based on Number of Subscribers")
        for _, row in top_courses.iterrows():
            course_title = row['course_title']
            course_url = row['url']
            course_price = row['price']
            num_subscribers = row['num_subscribers']
            stc.html(RESULT_TEMP.format(course_title, "", course_url, course_price, num_subscribers), height=250)

if __name__ == '__main__':
    main()
