import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_prep import get_clean_merged

# Load dataset
df = get_clean_merged()

# Build text representation for each book
df["book_text"] = (
    df["Book-Title"].fillna("") + " " +
    df["Book-Author"].fillna("") + " " +
    df.get("Summary", pd.Series([""] * len(df))).fillna("")
)

# Fit TF-IDF on book_text
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["book_text"])

def recommend_similar_books_by_text(title, top_n=5):
    df = get_clean_merged()

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["Book-Title"])

    if title.lower() not in df["Book-Title"].str.lower().values:
        return ["Book not found."]

    idx = df[df["Book-Title"].str.lower() == title.lower()].index[0]
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similarities[idx] = -1  # 👈 exclude the book itself

    top_indices = similarities.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][["Book-Title", "Book-Author"]].drop_duplicates()
    return results.head(top_n).to_dict(orient="records")
