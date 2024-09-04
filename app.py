import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load dataset
@st.cache_data
def load_data(data):
    try:
        df = pd.read_csv(data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

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
    idx = course_indices.get(title, None)
    if idx is None:
        return pd.DataFrame()  # Return an empty DataFrame if title not found
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = [i[1] for i in sim_scores[1:num_of_rec+1]]
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']].head(num_of_rec)
    return final_recommended_courses

# HTML template for displaying results with enhanced styling and icons
RESULT_TEMP = """
<div style="width:100%;height:100%;margin:5px;padding:10px;position:relative;border-radius:10px;
box-shadow:0 0 10px 2px #009688; background-color: #ffffff; border-left: 5px solid #009688; margin-bottom: 20px;">
<h4 style="color:#009688;">{}</h4>
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

# Function to filter courses by category
@st.cache_data
def filter_courses_by_category(df, category):
    if category:
        filtered_df = df[df['subject'] == category]
    else:
        filtered_df = df
    return filtered_df

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
        background-color: #009688;
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
        background-color: #009688 !important;
        color: white !portant;
    }
    .css-1aumxhk:hover {
        background-color: #009688 !important;
        color: white !important;
    }
    .st-expander-header:focus {
        background-color: #009688 !important;
        color: white !important;
    }
    .stAlert {
        background-color: #009688 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Course Recommendation App")
    st.markdown("Welcome to the **Ultimate Course Finder**! Discover the perfect courses tailored to your passions and goals.")
    
    # Sidebar Menu with Enhanced Icons and Features
    st.sidebar.title("🔍 Navigation")
    menu = ["🏠 Home", "🔍 Recommend", "📘 About", "📈 Statistics"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)
    
    # Additional Sidebar Features
    st.sidebar.header("📊 Filter by Category")
    top_categories = ["Development", "Business", "Finance", "Design", "Marketing"]
    selected_category = st.sidebar.selectbox("📚 Top Categories", top_categories)

    # State management for toggling recommendations visibility
    if 'show_recommendations' not in st.session_state:
        st.session_state['show_recommendations'] = False

    # Top Rated Courses button right below the menu with toggle functionality
    if 'show_top_rated' not in st.session_state:
        st.session_state['show_top_rated'] = False
    
    if st.sidebar.button("🎓 Top Rated Courses"):
        st.session_state['show_top_rated'] = not st.session_state['show_top_rated']

    # Load dataset
    df = load_data("data/udemy_course_data.csv")
    
    if df.empty:
        st.error("No data to display. Please check the dataset path and ensure the file is available.")
        return
    
    if choice == "🏠 Home":
        st.subheader("🏠 Home")
        st.markdown("Explore a curated selection of top courses from our extensive collection. Dive in and start learning today!")
        
        if selected_category:
            df_filtered = filter_courses_by_category(df, selected_category)
            if not df_filtered.empty:
                st.dataframe(df_filtered.head(10))
            else:
                st.warning("No courses available for the selected category.")
        else:
            st.dataframe(df.head(10))
    
    elif choice == "🔍 Recommend":
        st.subheader("🔍 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Enter Course Title: Discover courses that match your interests.")
        
        if st.button("Recommend"):
            st.session_state['show_recommendations'] = not st.session_state['show_recommendations']
        
        if st.session_state['show_recommendations']:
            if search_term:
                try:
                    num_of_rec = 10  # Default number of recommendations
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    st.markdown("### 🎯 Recommendations")
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.json(results_json)  # Removed color argument
                    
                    if not results.empty:
                        for _, row in results.iterrows():
                            rec_title = row['course_title']
                            rec_score = row['similarity_score']
                            rec_url = row['url']
                            rec_price = row['price']
                            rec_num_sub = row['num_subscribers']
                            stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)
                    else:
                        st.warning("No recommendations available for the provided course title.")
                
                except KeyError:
                    # Search for similar courses only if exact match is not found
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.info("Suggested Options:")
                        st.dataframe(result_df[['course_title', 'price', 'num_subscribers']])
                    else:
                        st.error("Course title not found. Please try a different search term.")
            else:
                st.warning("Please enter a course title to get recommendations.")
    
    elif choice == "📘 About":
        st.subheader("📘 About Us")
        st.markdown("Learn more about our mission and how we curate the best courses to help you succeed!")
    
    elif choice == "📈 Statistics":
        st.subheader("📈 Course Statistics")
        st.markdown("Explore various statistics about our courses and users.")
        
        if st.session_state['show_top_rated']:
            top_rated_df = get_top_rated_courses(df)
            st.markdown("### 🎓 Top Rated Courses")
            for _, row in top_rated_df.iterrows():
                rec_title = row['course_title']
                rec_url = row['url']
                rec_price = row['price']
                rec_num_sub = row['num_subscribers']
                st.markdown(f"**{rec_title}**\n💰 Price: {rec_price} | 👥 Students: {rec_num_sub}\n[Link]({rec_url})")

if __name__ == '__main__':
    main()
