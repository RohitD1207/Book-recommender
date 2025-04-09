# A Different Book System

A modular and extensible book recommendation system that combines collaborative filtering and content-based filtering techniques, developed using Python and Streamlit.

This project is designed for evaluating and demonstrating two primary recommendation approaches:

1. **SVD-based Collaborative Filtering** using scikit-learn
2. **TF-IDF-based Content Similarity** using natural language processing

It also includes functionality for assessing model performance through standard evaluation metrics.

---

## 📁 Project Structure

```
A different Book system/
├── src/
│   ├── app.py                 # Streamlit application
│   ├── data_prep.py           # Data cleaning and merging functions
│   ├── evaluate_model.py      # Evaluation logic for SVD and TF-IDF models
│   └── train_model.py         # Model training and encoder saving
├── models/
│   ├── svd_model.pkl          # Trained TruncatedSVD model
│   ├── user_encoder.pkl       # LabelEncoder for users
│   └── book_encoder.pkl       # LabelEncoder for books
├── data/
│   └── *.csv                  # Book-Crossing dataset files
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.10 or higher
- Required Python packages (see below)

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended libraries:
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `scipy`

---

### Step 1: Prepare the Data

Place the following datasets inside the `data/` directory:

- `Books.csv`
- `Users.csv`
- `Ratings.csv`

These files should follow the format from the Book-Crossing dataset.

---

### Step 2: Train the Model

Run the following script to train and save the SVD model and encoders:

```bash
python src/train_model.py
```

This will generate:
- Trained SVD model (`svd_model.pkl`)
- LabelEncoders for users and books

---

### Step 3: Launch the Streamlit App

To start the application locally:

```bash
streamlit run src/app.py
```

Then open the provided `localhost:8501` link in your browser.

---

## 📊 Evaluation Metrics

Two evaluation pipelines are available in `evaluate_model.py`:

### SVD-based Collaborative Filtering
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Precision, Recall, F1 Score (threshold: rating ≥ 6)

### TF-IDF-based Content Filtering
- Precision, Recall, F1 Score
- Evaluates whether top similar books share the same author

---

## 🔧 Technologies Used

- **Python** for scripting and logic
- **Streamlit** for the user interface
- **scikit-learn** for ML models and metrics
- **pandas / numpy / scipy** for data manipulation

---

## 📌 Future Enhancements

- Incorporate genre or keyword filtering
- Add user interaction within the app (ratings, feedback)
- Deploy via Streamlit Cloud or Hugging Face Spaces
- Introduce clustering for advanced personalization

---

## 📄 License

This project is developed as part of an academic exercise and is available for educational use. For reuse or deployment, please contact the author.

---

## 👤 Author

Rohit D  
B.Tech. in Computer Science & Engineering  
Specialization: Decision Science & Machine Learning  
Lovely Professional University
