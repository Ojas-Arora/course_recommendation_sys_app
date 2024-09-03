import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as stc

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

# Function to generate graphs
def generate_graphs(df):
    if 'price' not in df.columns or 'num_subscribers' not in df.columns or 'category' not in df.columns:
        st.error("The required columns ('price', 'num_subscribers', 'category') are missing in the dataset.")
        return None, None, None, None
    
    # Distribution of Course Prices
    price_histogram = px.histogram(df, x='price', nbins=20, title='Distribution of Course Prices')
    
    # Number of Subscribers by Course
    subscribers_bar_chart = px.bar(df, x='course_title', y='num_subscribers', title='Number of Subscribers by Course', height=400)

    # Course Count by Category
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Course Count']
    category_pie_chart = px.pie(category_counts, names='Category', values='Course Count', title='Course Count by Category')

    # Average Course Price by Category
    avg_price_by_category = df.groupby('category')['price'].mean().reset_index()
    avg_price_by_category.columns = ['Category', 'Average Price']
    avg_price_bar_chart = px.bar(avg_price_by_category, x='Category', y='Average Price', title='Average Course Price by Category')

    return price_histogram, subscribers_bar_chart, category_pie_chart, avg_price_bar_chart

# HTML template for displaying results with enhanced styling and icons
RESULT_TEMP = """
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

    # Load dataset
    df = load_data("data/udemy_course_data.csv")

    # Display column names to debug
    st.write("Columns in the dataset:")
    st.write(df.columns)

    menu = ["Home", "Recommend", "Graphs", "About"]
    choice = st.sidebar.selectbox("Menu", menu, index=0)

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
                        stc.html(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), height=250)

                except KeyError:
                    # Search for similar courses only if exact match is not found
                    result_df = search_term_if_not_found(search_term, df)
                    if not result_df.empty:
                        st.info("Suggested Options:")
                        st.dataframe(result_df)
                    else:
                        st.warning("Course not found. Please try a different search term.")
    
    elif choice == "Graphs":
        st.subheader("📊 Graphs")
        price_histogram, subscribers_bar_chart, category_pie_chart, avg_price_bar_chart = generate_graphs(df)
        if price_histogram:
            st.plotly_chart(price_histogram)
            st.plotly_chart(subscribers_bar_chart)
            st.plotly_chart(category_pie_chart)
            st.plotly_chart(avg_price_bar_chart)
    
    else:
        st.subheader("ℹ️ About")
        st.markdown("This app is built using Streamlit and Pandas to demonstrate a basic course recommendation system.")

if __name__ == '__main__':
    main()
