import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# Set Page config
st.set_page_config(page_title="Book Recommendation", layout="wide")

# Custom CSS Styling
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Merriweather&display=swap" rel="stylesheet">

    <style>
        html, body, [class^="css"] {
            font-family: 'Merriweather', serif !important;
        }

        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url("https://images.unsplash.com/photo-1532012197267-da84d127e765");
            background-size: cover;
            background-position: center;
            opacity: 0.9;
            z-index: -1;
        }

        .stApp {
            background-color: rgba(249, 243, 239, 0.85);
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Merriweather', serif !important;
            color: #4c2b08;
            text-align: center;
        }

        .stButton > button {
            background-color: #ab7743;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #6d3914;
        }

        .stSelectbox > div {
            background-color: #ffffff;
            color: #4c2b08;
            border-radius: 8px;
            padding: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Header
st.markdown("<h1>📚 Book Recommendation System</h1>", unsafe_allow_html=True)

# Function to fetch posters and ratings
def fetch_info(suggestion):
    book_names_list = []
    poster_urls = []
    ratings = []

    for book_id in suggestion[0]:
        name = book_pivot.index[book_id]
        book_names_list.append(name)

        # Find corresponding row in final_rating
        matches = final_rating[final_rating['title'] == name]
        if not matches.empty:
            poster_url = matches.iloc[0]['image_url']
            avg_rating = matches['rating'].mean() / 2  # Convert 10 to 5
        else:
            poster_url = "https://via.placeholder.com/150"
            avg_rating = 0

        poster_urls.append(poster_url)
        ratings.append(round(avg_rating, 1))  # 1 decimal point

    return book_names_list, poster_urls, ratings

# Recommendation logic
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    return fetch_info(suggestion)

# Select box
selected_book = st.selectbox("📖 Type or select a book from the dropdown", book_names)

# Show recommendations
if st.button("Show Recommendation"):
    titles, images, ratings = recommend_book(selected_book)

    cols = st.columns(5)
    for i in range(1, 6):  # Skipping index 0 as it's the same book
        with cols[i - 1]:
            st.markdown(f"""
                <div style="
                    background-color: #d7bda6;
                    border-radius: 16px;
                    padding: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    text-align: center;
                    height: 100%;
                ">
                    <img src="{images[i]}" style="width: 100%; height: 200px; object-fit: contain; border-radius: 8px;" />
                    <h4 style="color: #4c2b08; margin-top: 10px; font-size: 16px;">{titles[i]}</h4>
                    <div style="color: #6d3914; font-weight: bold;">⭐ {ratings[i]} / 5</div>
                </div>
            """, unsafe_allow_html=True)
