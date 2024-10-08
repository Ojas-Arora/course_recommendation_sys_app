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
box-shadow:0 0 10px 2px #191970; background-color: #ffffff; border-left: 5px solid #191970; margin-bottom: 20px; color:#191970;">
<h4 style="color:#191970;">{}</h4>
<p style="color:#191970;"><span style="color:#333;">🔍 Similarity Score:</span> {}</p>
<p style="color:#191970;"><span style="color:#333;">🔗</span> <a href="{}" target="_blank">Course Link</a></p>
<p style="color:#191970;"><span style="color:#333;">💰 Price:</span> {}</p>
<p style="color:#191970;"><span style="color:#333;">👥 Students Enrolled:</span> {}</p>
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
        background-color: #191970;
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
    .stButton > button {
    background-color: #191970 ; /* Button background color */
    color: white; /* Button text color */
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #191970; /* Hover color */
}           

    /* Change button and box active color */
    button:active, .stButton>button:focus {
        background-color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    .css-1aumxhk:hover {
        background-color: #191970 !important;
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }
    .header h2 {
        color: #191970;
        animation: fadeIn 3s ease-in;
    }
    .st-expander-header:focus {
        background-color: #191970 !important;
        color: #191970 !important;  /* Changed text color */
        animation: fadeIn 2s ease-in; /* Added animation */
    }

    .stAlert {
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
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
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
        search_term = st.text_input("""
""")
        st.markdown("""
<style>
.stButton > button {
    background-color: #191970 ; /* Button background color */
    color: white; /* Button text color */
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #191970; /* Hover color */
}
</style>
""", unsafe_allow_html=True)
        if st.button("Recommend"):
            st.session_state['show_recommendations'] = not st.session_state['show_recommendations']
        
        if st.session_state['show_recommendations']:
            if search_term:
                try:
                    num_of_rec = 10  # Default number of recommendations
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    st.markdown(
                    f"<h3 style='color:#191970;'>🎯 Recommendations</h3>",
                    unsafe_allow_html=True
                )
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.json(results_json)
                    
                    st.markdown(
                    f"<h3 style='color:#191970;'>📘 Recommended Courses</h3>",
                    unsafe_allow_html=True
                )
                    for _, row in results.iterrows():
                        stc.html(RESULT_TEMP.format(row['course_title'], row['similarity_score'], row['url'], row['price'], row['num_subscribers']), height=250)
                
                except KeyError:
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.dataframe(result_df)
                    else:
                       st.markdown(
                    f"<p style='color:#191970;'>No results found for '{search_term}'. Try another title.</p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
            "<p style='color:#191970;'>Please enter a course title.</p>",
            unsafe_allow_html=True
        )
    
    elif choice == "📘 About":
        st.markdown(
        '<h3 style="color:#191970;">📘 About This App</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
    <style>
    .custom-header {
        color: #191970;
    }
    .h2 {
        color: #191970;
        animation: fadeIn 3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    <h2 class="h2">Welcome to the <strong>Course Recommendation App🚀</strong></h2>

    <h3 class="custom-header">🎯 <strong>Objective:</strong></h3>
    <p>This app is designed to help you discover the best courses that match your learning interests. With an extensive collection of courses, our goal is to provide personalized recommendations to guide your educational journey.</p>

    <h3 class="custom-header">🔍 <strong>Features:</strong></h3>
    <ul>
        <li><strong>📚 Course Recommendations</strong>: Get personalized course suggestions based on the title you provide. Our system uses advanced text vectorization and similarity measures to find the most relevant courses for you.</li>
        <li><strong>🌟 Top Rated Courses</strong>: Explore the most popular courses based on student enrollment and price. We showcase top-rated options to help you make informed decisions.</li>
        <li><strong>📊 Detailed Statistics</strong>: Access in-depth statistics about course popularity, pricing, and student engagement to better understand market trends.</li>
    </ul>

    <h3 class="custom-header">🛠️ <strong>Technology Stack:</strong></h3>
    <ul>
        <li><strong>🔧 Backend</strong>: Python with Streamlit for the web framework.</li>
        <li><strong>🔢 Text Vectorization</strong>: <code>CountVectorizer</code> from Scikit-learn to convert course titles into numerical data.</li>
        <li><strong>🔍 Similarity Computation</strong>: <code>cosine_similarity</code> from Scikit-learn to find similarity between courses.</li>
        <li><strong>📈 Data Handling</strong>: Pandas for data manipulation and analysis.</li>
    </ul>

    <h3 class="custom-header">⚙️ <strong>How It Works:</strong></h3>
    <ol>
        <li><strong>📥 Upload Data</strong>: The app reads course data from a CSV file.</li>
        <li><strong>🔄 Vectorize Text</strong>: It converts course titles into numerical vectors.</li>
        <li><strong>📐 Compute Similarity</strong>: It calculates the cosine similarity between course titles.</li>
        <li><strong>🎯 Provide Recommendations</strong>: Based on your search, it provides a list of recommended courses.</li>
    </ol>
""", unsafe_allow_html=True)
    elif choice == "📈 Statistics":
        st.markdown(
        '<h3 style="color:#191970;">📈 Statistics</h3>',
        unsafe_allow_html=True
    )
        st.markdown("""
    <style>
    .custom-header {
        color: #191970;
    }
    </style>
    <p>Explore detailed statistics and trends on course popularity, pricing, and student enrollment. 📊</p>

    <h3 class="custom-header">🔍 <strong>What You'll Find:</strong></h3>
    <ul>
        <li><strong>📈 Course Popularity:</strong> Discover which courses are trending based on student reviews and enrollment numbers.</li>
        <li><strong>💰 Pricing Insights:</strong> Analyze pricing patterns to find courses that offer the best value for your investment.</li>
        <li><strong>👥 Student Enrollment:</strong> Understand enrollment trends to gauge course demand and popularity.</li>
    </ul>

    <h3 class="custom-header">🛠️ <strong>How This Helps You:</strong></h3>
    <ul>
        <li><strong>📈 Make Informed Choices:</strong> Use popularity trends to select courses that are in demand.</li>
        <li><strong>💵 Optimize Spending:</strong> Evaluate pricing trends to budget effectively for your learning.</li>
        <li><strong>📚 Enhance Learning Path:</strong> Leverage student enrollment data to choose courses with high engagement and effectiveness.</li>
    </ul>

    <p>📥 Use this data to make informed decisions about your learning path. Whether you're looking for the most popular courses or seeking the best deals, our statistics provide valuable insights to guide your choices.</p>

    <p>🔍 Dive into the data and enhance your educational journey with the knowledge you need to succeed!</p>
""", unsafe_allow_html=True)
        top_rated_df = get_top_rated_courses(df)
        st.dataframe(top_rated_df)
    
    # Toggle for Top Rated Courses in Sidebar
    if st.session_state['show_top_rated']:
        top_rated_df = get_top_rated_courses(df)
        st.sidebar.markdown("### 🎓 Top Rated Courses")
        for _, row in top_rated_df.iterrows():
            rec_title = row['course_title']
            rec_url = row['url']
            rec_price = row['price']
            rec_num_sub = row['num_subscribers']
            st.sidebar.markdown(f"**{rec_title}**\n💰 Price: {rec_price} | 👥 Students: {rec_num_sub}\n[Link]({rec_url})")

if __name__ == '__main__':
    main()
