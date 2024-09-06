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
    
    # Manually set the prices for the top 10 courses
    prices = [549, 799, 799, 649, 649, 549, 649, 649, 649, 549]
    top_rated_df['price'] = prices
    
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
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Existing Styles */
    .main {
        background-color: rgb(250,235,215);
        color: #191970;  /* Changed text color to #191970 */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    .css-1f3v6nr {
        background-position: center;
        color: #191970;  /* Changed text color to #191970 */
        background-color: #191970 !important;  /* Changed background color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    /* Custom styling for sidebar */
    .css-1d391kg {
        background-color: #009688;
        color: #191970;  /* Changed text color to #191970 */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    /* Custom styling for the menu items */
    .css-1n1n7f2 {
        padding: 10px;
        border-radius: 10px;
        color: #191970 !important;  /* Changed text color */
        background-color: #191970 !important;  /* Changed background color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    /* Styling for the header */
    .css-1d391kg h1 {
        color: #191970;
        background-color: #191970 !important;  /* Changed background color */
        animation: fadeIn 3s ease-in; /* Added animation */
    }

    /* Change button and box active color */
    button:active, .stButton>button:focus {
        background-color: #009688 !important;
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    .css-1aumxhk:hover {
        background-color: #3b8c88 !important;
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }
    .header h2 {
        color: #191970;
        animation: fadeIn 3s ease-in;
    }
    .st-expander-header:focus {
        background-color: #3b8c88 !important;
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    .stAlert {
        background-color: #191970 !important;  /* Changed background color */
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }
</style>

    """, unsafe_allow_html=True)

    # Sidebar Menu with Enhanced Icons and Features
    st.sidebar.title("🔍 Navigation")
    menu = ["🏠 Home", "🔍 Recommend", "📘 About", "📈 Statistics"]
    choice = st.sidebar.selectbox(" 📚 Menu", menu, index=0)
    st.sidebar.markdown("""
    <style>
    .css-1n1n7f2, .css-1n1n7f2 * {
        color: #191970 !important;
        background-color: #191970 !important;   
        animation: fadeIn 2s ease-in; /* Added animation */             
    }
    .menu{
        color: #191970 !important;
        background-color: #191970 !important;
         animation: fadeIn 2s ease-in; /* Added animation */
    }
    </style>
    """, unsafe_allow_html=True)

    # Quick Stats in Sidebar
    st.sidebar.header("📊 Quick Stats")
    st.sidebar.metric(" 🌟 Total Courses", "500+")

    # Load dataset
    df = load_data("data/udemy_course_data.csv")
    
    # State management for toggling recommendations visibility
    if 'show_recommendations' not in st.session_state:
        st.session_state['show_recommendations'] = False

    # Top Rated Courses button right below the menu with toggle functionality
    if 'show_top_rated' not in st.session_state:
        st.session_state['show_top_rated'] = False
    
    if st.sidebar.button("🎓 Top Rated Courses"):
        st.session_state['show_top_rated'] = not st.session_state['show_top_rated']

    if choice == "🏠 Home":
        st.markdown(
        '<h3 style="color:#191970;">🏠 Home</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
    <style>
    .css-1vhystk p, .css-1vhystk ul, .css-1vhystk li, .css-1vhystk h3 {
        color: #191970 !important;
        animation: fadeIn 2s ease-in; /* Added animation */
    }
    .choice{
        color: #191970 !important;    
        animation: fadeIn 2s ease-in; /* Added animation */       
    }
    </style>
    """, unsafe_allow_html=True)
        st.markdown( """ 
        <div class='header'>
        <h2> 🌟 Explore Top Courses</h2>                        
🎓 Discover a curated selection of top courses from our extensive collection. With our handpicked recommendations, you can:

- **📚 Explore Quality Courses**: Access a variety of high-quality courses tailored to different interests and skill levels.
- **🌟 Find Top Picks**: Dive into the most popular and highly-rated courses available.
- **🚀 Start Learning Today**: Begin your educational journey with courses that are designed to enhance your skills and knowledge.

🔢Don’t miss out on the opportunity to learn from the best. Start exploring now and take the next step in your learning adventure! 🌐
        </div>
        """, unsafe_allow_html=True)
    
        st.dataframe(df.head(10))
    
    elif choice == "🔍 Recommend":
        st.markdown(
        '<h3 style="color:#191970;">🔍 Recommend Courses</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
        <style>
        .custom-header {
            color: #191970;
        }
        .custom-description {
            color: #191970;
            font-size: 18px; /* Adjust font size as needed */
        }
        </style>
        <h3 class="custom-header">📐 Enter Course Title</h3>
        <p class="custom-description">🧠 Discover courses that align with your interests.<br></br> 
        📚 Type in a course title to get personalized recommendations tailored just for you</p>
    """, unsafe_allow_html=True)
    
    # Text input widget
    search_term = st.text_input(
        label="",
        placeholder="🔍 Search for a course to get customized recommendations just for you! 🚀"
    )
    
    # Button to toggle recommendations
    if st.button("Recommend"):
        if 'show_recommendations' not in st.session_state:
            st.session_state['show_recommendations'] = False
        st.session_state['show_recommendations'] = not st.session_state['show_recommendations']
    
    # Handle recommendations display
    if 'show_recommendations' in st.session_state and st.session_state['show_recommendations']:
        if search_term:
            try:
                cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
                num_of_rec = 10  # Default number of recommendations
                results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                st.markdown("### 🎯 Recommendations")
                
                with st.expander("Results as JSON"):
                    results_json = results.to_dict('index')
                    st.json(results_json)  # Display results as JSON
                    
                # Display recommendations
                for _, row in results.iterrows():
                    rec_title = row['course_title']
                    rec_score = row['similarity_score']
                    rec_url = row['url']
                    rec_price = row['price']
                    rec_num_sub = row['num_subscribers']
                    stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)
            
            except KeyError:
                # Handle the case where the key is not found
                result_df = search_term_if_not_found(search_term)
                st.markdown("### 📉 No Results Found")
                st.write(result_df)
    
    # Top Rated Courses Display
    if st.session_state.get('show_top_rated', False):
        st.markdown("### 📈 Top Rated Courses")
        top_rated_courses = get_top_rated_courses(df)
        st.dataframe(top_rated_courses)
    
    # About Page
    elif choice == "📘 About":
        st.markdown(
        '<h3 style="color:#191970;">📘 About This App</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
    <style>
    .about-header {
        color: #191970;
    }
    .about-description {
        color: #191970;
        font-size: 18px; /* Adjust font size as needed */
    }
    </style>
    <h3 class="about-header">📘 About Course Recommendation App</h3>
    <p class="about-description">This app helps users find the best courses based on their interests and preferences. You can get personalized recommendations or explore top-rated courses.</p>
    """, unsafe_allow_html=True)
    
    # Statistics Page
    elif choice == "📈 Statistics":
        st.markdown(
        '<h3 style="color:#191970;">📈 Statistics</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
    <style>
    .stats-header {
        color: #191970;
    }
    .stats-description {
        color: #191970;
        font-size: 18px; /* Adjust font size as needed */
    }
    </style>
    <h3 class="stats-header">📈 Statistics Overview</h3>
    <p class="stats-description">Here you can view various statistics related to course recommendations and user preferences. The data is dynamically updated based on user interactions.</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()