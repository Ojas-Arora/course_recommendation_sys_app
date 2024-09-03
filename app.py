# Import necessary libraries
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
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses

# HTML template for displaying results with light and dark mode styles
RESULT_TEMP_LIGHT = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:10px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #f0f0f0;
border-left: 5px solid #6c6c6c; margin-bottom: 20px;">
<h4 style="color:#333;">{}</h4>
<p style="color:#0073e6;"><span style="color:#333;">📈 Similarity Score:</span> {}</p>
<p style="color:#0073e6;"><span style="color:#333;">🔗</span> <a href="{}" target="_blank">Course Link</a></p>
<p style="color:#0073e6;"><span style="color:#333;">💲 Price:</span> {}</p>
<p style="color:#0073e6;"><span style="color:#333;">🧑‍🎓 Students Enrolled:</span> {}</p>
</div>
"""

RESULT_TEMP_DARK = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:10px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #333; background-color: #333;
border-left: 5px solid #6c6c6c; margin-bottom: 20px;">
<h4 style="color:#fff;">{}</h4>
<p style="color:#00ace6;"><span style="color:#fff;">📈 Similarity Score:</span> {}</p>
<p style="color:#00ace6;"><span style="color:#fff;">🔗</span> <a href="{}" target="_blank" style="color:#00ace6;">Course Link</a></p>
<p style="color:#00ace6;"><span style="color:#fff;">💲 Price:</span> {}</p>
<p style="color:#00ace6;"><span style="color:#fff;">🧑‍🎓 Students Enrolled:</span> {}</p>
</div>
"""

# Function to search term if not found
@st.cache_data
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False)]
    return result_df

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Course Recommendation App", page_icon="🎓")
    st.title("🎓 Course Recommendation App")
    st.markdown("Welcome to the **Course Recommendation App**! Find courses tailored to your interests.")
    
    # Toggle for light/dark mode
    mode = st.sidebar.radio("Select Mode", ["Light", "Dark"])

    # Set style based on mode
    if mode == "Light":
        result_template = RESULT_TEMP_LIGHT
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        result_template = RESULT_TEMP_DARK
        st.markdown(
            """
            <style>
            body {
                background-color: #0E1117;
                color: #ffffff;
            }
            </style>
            """, unsafe_allow_html=True
        )
    
    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)
    
    # Load dataset
    df = load_data("data/udemy_course_data.csv")

    if choice == "Home":
        st.subheader("🏠 Home")
        st.markdown("Browse the first few courses from our dataset:")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
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
                        stc.html(result_template.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)

                except KeyError:
                    # Search for similar courses only if exact match is not found
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.info("Suggested Options:")
                        st.dataframe(result_df)
                    else:
                        st.warning("Course not found. Please try a different search term.")
    else:
        st.subheader("ℹ️ About")
        st.markdown("This app is built using Streamlit and Pandas to demonstrate a basic course recommendation system.")

if __name__ == '__main__':
    main()
