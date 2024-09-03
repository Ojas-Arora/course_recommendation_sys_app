# Import necessary libraries
import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import json

# Function to load Lottie animations from a JSON file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load a Lottie animation for visual appeal
lottie_animation = load_lottiefile("path_to_lottiefile.json")  # Update with your JSON file path

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
@st.cache
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

# Enhanced HTML template for displaying results with more styling
RESULT_TEMP = """
<div style="width: 90%; height: 100%; margin: 10px; padding: 10px; position: relative; border-radius: 15px; 
            box-shadow: 0 0 15px 5px rgba(0, 0, 0, 0.2); background-color: #f4f4f4; border-left: 5px solid #6c6c6c;">
    <h4 style="color: #ff4b4b;">🔍 {}</h4>
    <p style="color: blue;"><span style="color: black;">📈 Similarity Score:</span> {}</p>
    <p style="color: blue;"><span style="color: black;">🔗 Course URL:</span> <a href="{}" target="_blank">Visit Course</a></p>
    <p style="color: blue;"><span style="color: black;">💲 Price:</span> {}</p>
    <p style="color: blue;"><span style="color: black;">👨‍🎓 Number of Students:</span> {}</p>
</div>
"""

# Function to search term if not found
@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False)]
    return result_df

# Main function for Streamlit app
def main():
    st.title("🎓 Course Recommendation App")
    
    # Display Lottie animation at the top
    st_lottie(lottie_animation, height=200, key="animation")

    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load data
    df = load_data("data/udemy_course_data.csv")

    if choice == "Home":
        st.subheader("🏠 Home")
        st.write("Welcome to the Course Recommendation App! Explore top courses tailored to your interests.")
        st.dataframe(df.head(10))

    elif choice == "Recommend":
        st.subheader("📚 Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Enter a course title to search:")
        num_of_rec = st.sidebar.number_input("Number of Recommendations", 4, 30, 7)
        
        if st.button("🔍 Recommend"):
            if search_term:
                try:
                    results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                    with st.expander("📝 Results as JSON"):
                        results_json = results.to_dict('index')
                        st.json(results_json)

                    for _, row in results.iterrows():
                        rec_title = row['course_title']
                        rec_score = row['similarity_score']
                        rec_url = row['url']
                        rec_price = row['price']
                        rec_num_sub = row['num_subscribers']
                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=200)
                except KeyError:
                    st.warning("⚠️ Course not found. Try a different search term.")
                    st.info("💡 Suggested Options:")
                    result_df = search_term_if_not_found(search_term, df)
                    st.dataframe(result_df)
    
    else:
        st.subheader("ℹ️ About")
        st.markdown("""
        This app recommends online courses based on your search term using a content-based recommendation system.
        - **Libraries used**: Streamlit, Pandas, Scikit-learn, Plotly
        - **Data**: Sourced from Udemy course dataset
        """)
        
if __name__ == '__main__':
    main()
