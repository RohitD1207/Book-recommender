import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from hybrid_recommender import recommend_books_for_user, recommend_similar_books
from tfidf_recommender import recommend_similar_books_by_text
from evaluate_model import get_svd_metrics

st.set_page_config(page_title="📚 Book Recommender", layout="centered")
st.title("📖 Book Recommender")
st.markdown("Get recommendations *based on users or book titles.*")

# --- Tabs
tab1, tab2, tab3 = st.tabs(["🔑 By User", "📘 By Book", "📊 Model Evaluation"])

with tab1:
    user_id = st.number_input("Enter User ID", min_value=0, value=276729)
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Get Recommendations", key="user"):
        with st.spinner("Cooking up your books..."):
            results = recommend_books_for_user(user_id, top_n)
        st.subheader("Recommended Books:")
        if isinstance(results[0], dict):
            for book in results:
                st.markdown(f"- **{book['Book-Title']}** by *{book['Book-Author']}*")
        else:
            st.error(results[0])

with tab2:
    book_title = st.text_input("Enter Book Title", value="Harry Potter and the Philosopher's Stone")
    method = st.selectbox("Choose recommendation method", ["SVD (user-based)", "TF-IDF (content-based)"])
    top_n = st.slider("Number of similar books", 1, 10, 5, key="bookslider")

    if st.button("Find Similar Books", key="book"):
        with st.spinner("Digging through the shelves..."):
            if method == "SVD (user-based)":
                results = recommend_similar_books(book_title, top_n)
            else:
                results = recommend_similar_books_by_text(book_title, top_n)

        st.subheader("Similar Books:")
        if isinstance(results[0], dict):
            for book in results:
                st.markdown(f"- **{book['Book-Title']}** by *{book['Book-Author']}*")
        else:
            st.error(results[0])

with tab3:
    st.subheader("📊 Evaluation Results")
    
    st.markdown("### 🔷 SVD Recommender")
    svd_metrics = get_svd_metrics()
    for k, v in svd_metrics.items():
        st.markdown(f"- **{k}**: {v:.4f}")
