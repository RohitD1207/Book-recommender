import pandas as pd

def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    books = pd.read_csv("data/books.csv")
    return ratings, books

def merge_data():
    ratings, books = load_data()
    merged = ratings.merge(books, on="ISBN")
    return merged

def get_ratings_only():
    ratings, _ = load_data()
    return ratings[["User-ID", "ISBN", "Book-Rating"]]

def get_books_only():
    _, books = load_data()
    return books[["ISBN", "Book-Title", "Book-Author"]]

def get_clean_merged():
    merged = merge_data()
    merged = merged[["User-ID", "ISBN", "Book-Rating", "Book-Title", "Book-Author"]]
    return merged
