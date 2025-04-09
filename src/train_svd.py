import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from data_prep import get_clean_merged

# Load merged dataset
df = get_clean_merged()

# Encode User and Book IDs
user_enc = LabelEncoder()
book_enc = LabelEncoder()

df["user_idx"] = user_enc.fit_transform(df["User-ID"])
df["book_idx"] = book_enc.fit_transform(df["ISBN"])

# Create user-item matrix (sparse)
user_book_matrix = csr_matrix(
    (df["Book-Rating"], (df["user_idx"], df["book_idx"]))
)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(user_book_matrix)
book_features = svd.components_.T  # (n_books, n_components)

# Save everything
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

with open("models/user_encoder.pkl", "wb") as f:
    pickle.dump(user_enc, f)

with open("models/book_encoder.pkl", "wb") as f:
    pickle.dump(book_enc, f)

with open("models/user_features.pkl", "wb") as f:
    pickle.dump(user_features, f)

# Save shape info (optional, for sanity checks later)
with open("models/matrix_shape.pkl", "wb") as f:
    pickle.dump(user_book_matrix.shape, f)

print("✅ Training complete. Everything saved in /models")
