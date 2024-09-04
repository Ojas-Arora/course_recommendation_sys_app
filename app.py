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
    selected_course_indices = [i[0] for i in sim_scores[1:num_of_rec+1]]  # Corrected to ensure exact number of recommendations
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = [i[1] for i in sim_scores[1:num_of_rec+1]]
    final_recommended_courses = result_df[['course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses

# HTML template for displaying results
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

# Function to search term if not found
@st.cache_data
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term, case=False)]
    return result_df

# Main function for Streamlit app
def main():
    # Set page config at the start
    st.set_page_config(page_title="Course Recommendation App", page_icon="🎓")

    # Load dataset
    df = load_data("data/udemy_course_data.csv")

    # Sidebar Menu
    st.sidebar.title("Menu")
    st.sidebar.subheader("Number of Recommendations")
    num_of_rec = st.sidebar.slider("Select number", 4, 30, 7)
    
    # Recommendation section
    st.subheader("🔍 Recommend Courses")
    cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
    search_term = st.text_input("Enter Course Title")

    if st.button("Recommend"):
        if search_term:
            try:
                results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                st.markdown("### 🎯 Recommendations")
                for _, row in results.iterrows():
                    rec_title = row['course_title']
                    rec_score = row['similarity_score']
                    rec_url = row['url']
                    rec_price = row['price']
                    rec_num_sub = row['num_subscribers']
                    stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)
            except KeyError:
                result_df = search_term_if_not_found(search_term, df)
                if not result_df.empty:
                    st.info("Suggested Options:")
                    st.dataframe(result_df)
                else:
                    st.warning("Course not found. Please try a different search term.")

if __name__ == "__main__":
    main()
