import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from data_prep import get_clean_merged


# --- SVD METRICS ---
def get_svd_metrics():
    df = get_clean_merged()

    with open("models/svd_model.pkl", "rb") as f:
        svd = pickle.load(f)

    with open("models/user_encoder.pkl", "rb") as f:
        user_enc = pickle.load(f)

    with open("models/book_encoder.pkl", "rb") as f:
        book_enc = pickle.load(f)

    df = df.copy()
    df["user_idx"] = user_enc.transform(df["User-ID"])
    df["book_idx"] = book_enc.transform(df["ISBN"])

    user_book_matrix = csr_matrix(
        (df["Book-Rating"], (df["user_idx"], df["book_idx"])),
        shape=(len(user_enc.classes_), len(book_enc.classes_))
    )

    user_features = svd.transform(user_book_matrix)
    book_features = svd.components_.T

    # 💥 Memory-safe prediction using element-wise dot product
    pred_ratings = np.sum(
        user_features[df["user_idx"]] * book_features[df["book_idx"]],
        axis=1
    )

    true_ratings = df["Book-Rating"].values

    mse = mean_squared_error(true_ratings, pred_ratings)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_ratings, pred_ratings)

    true_binary = (true_ratings >= 6).astype(int)
    pred_binary = (pred_ratings >= 6).astype(int)

    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)

    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }


'''
# --- TF-IDF METRICS ---
def get_tfidf_metrics():
    df = get_clean_merged()

    df["book_text"] = (
        df["Book-Title"].fillna("") + " " +
        df["Book-Author"].fillna("") + " " +
        df.get("Summary", pd.Series([""] * len(df))).fillna("")
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["book_text"])

    cosine_sim = cosine_similarity(tfidf_matrix)

    # Ground truth: a book is "similar" to itself
    true = []
    pred = []

    for i in range(len(df)):
        top_n = cosine_sim[i].argsort()[::-1][1:6]  # Skip itself
        for j in top_n:
            true.append(1 if df.iloc[i]["Book-Author"] == df.iloc[j]["Book-Author"] else 0)
            pred.append(1)

    precision = precision_score(true, pred, zero_division=0)
    recall = recall_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred, zero_division=0)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Note": "TF-IDF is similarity-based, so no RMSE/MSE/MAE here",
    }
'''