# Readme.md

## App Platform

Web

## Tech Stack

- Flask
- Pandas
- Scikit-learn
- Faiss
- HTML/CSS

## App Intro

**Smart Recipe Finder**: A web-based application that helps users discover recipes based on the ingredients they have.

## Tool Intro

This app was built using:

- [Faiss](https://faiss.ai/) for efficient similarity search
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Flask](https://flask.palletsprojects.com/) for web application framework
- [Scikit-learn](https://scikit-learn.org/) for text preprocessing and dimensionality reduction

## Functionality Overview

The **Smart Recipe Finder** allows users to input a list of ingredients they have on hand. The app then leverages machine learning to find the top 5 most relevant recipes from a large dataset. Users are presented with recipe details, including a link to the full recipe on its source website.

### Key Features

- **Ingredient-Based Search**: Users input available ingredients, and the app finds recipes that match those inputs.
- **Efficient Recipe Matching**: Uses TF-IDF vectorization and Faiss for fast and accurate similarity search.
- **Dynamic Recipe Recommendations**: Presents top 5 most relevant recipes based on cosine similarity.

## Development Process

1. **Data Preprocessing**: The raw recipe dataset was cleaned and processed. Ingredients were tokenized, and a TF-IDF matrix was computed.
2. **Dimensionality Reduction**: Applied Truncated SVD to reduce dimensionality, optimizing for speed and accuracy.
3. **Similarity Search**: Faiss was used to efficiently find the most similar recipes in the dataset based on user input.
4. **Web Development**: Built a user-friendly interface using Flask for backend and HTML/CSS for frontend. Users input ingredients via a simple form, and results are displayed dynamically.

### Value

This app offers a fast and efficient way for users to find recipes that suit their available ingredients, minimizing food waste and enhancing meal planning. It demonstrates how machine learning techniques like vectorization and similarity search can be integrated into a practical, real-world application.

### Link to the Dataset:

https://www.kaggle.com/datasets/paultimothymooney/recipenlg/data
