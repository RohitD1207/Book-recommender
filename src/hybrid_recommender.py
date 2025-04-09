import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from data_prep import get_clean_merged

# Load all pickled components
with open("models/svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

with open("models/user_encoder.pkl", "rb") as f:
    user_enc = pickle.load(f)

with open("models/book_encoder.pkl", "rb") as f:
    book_enc = pickle.load(f)

with open("models/user_features.pkl", "rb") as f:
    user_features = pickle.load(f)

# Load the original dataset
df = get_clean_merged()

# Book features from trained SVD
book_features = svd.components_.T  # shape: (n_books, n_components)

# --------------------------
# 🔍 Recommend for a User
# --------------------------
def recommend_books_for_user(user_id, top_n=5):
    if user_id not in user_enc.classes_:
        return ["User not found."]

    user_idx = user_enc.transform([user_id])[0]
    user_vector = user_features[user_idx].reshape(1, -1)

    sims = cosine_similarity(user_vector, book_features).flatten()

    # Exclude books the user has already rated
    rated_books = df[df["User-ID"] == user_id]["ISBN"].values
    rated_book_idxs = book_enc.transform(rated_books)
    sims[rated_book_idxs] = -1

    top_indices = sims.argsort()[::-1][:top_n]
    top_isbns = book_enc.inverse_transform(top_indices)

    results = df[df["ISBN"].isin(top_isbns)][["Book-Title", "Book-Author"]].drop_duplicates()
    return results.head(top_n).to_dict(orient="records")

# --------------------------
# 🔍 Recommend Similar Books
# --------------------------
def recommend_similar_books(title, top_n=5):
    isbn = df[df["Book-Title"].str.lower() == title.lower()]["ISBN"].values
    if len(isbn) == 0:
        return ["Book not found."]
    
    book_idx = book_enc.transform([isbn[0]])[0]
    book_vector = book_features[book_idx].reshape(1, -1)

    sims = cosine_similarity(book_vector, book_features).flatten()
    sims[book_idx] = -1

    top_indices = sims.argsort()[::-1][:top_n]
    top_isbns = book_enc.inverse_transform(top_indices)

    results = df[df["ISBN"].isin(top_isbns)][["Book-Title", "Book-Author"]].drop_duplicates()
    return results.head(top_n).to_dict(orient="records")
