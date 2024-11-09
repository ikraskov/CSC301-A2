from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
import faiss

app = Flask(__name__)

df = pd.read_csv('quarter_recipe_dataset.csv')
# Lets work with a quarter of the dataset as the original dataset is too large
df = df.sample(frac=0.25, random_state=42)
df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(eval(x)))


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])


svd = TruncatedSVD(n_components=100) # this is done so that our matrix is not too sparse, and we can use cosine similarity faster
reduced_matrix = svd.fit_transform(tfidf_matrix)
reduced_matrix = reduced_matrix.astype('float32') # Reduce memory usage by using float32 instead of float64 and make it compatible with Faiss

# Faiss for approximate nearest neighbors
index = faiss.IndexFlatL2(reduced_matrix.shape[1]) # Construct the index using L2 distance 
faiss.normalize_L2(reduced_matrix)  # Normalize vectors for cosine 
index.add(reduced_matrix) # this line adds the vectors to the index which makes searching faster and more efficient

def find_similar_recipes(user_ingredients, top_n=5):
    ingredients = ' '.join(user_ingredients)
    user_vector = vectorizer.transform([ingredients])
    user_input_svd = svd.transform(user_vector)
    user_input_svd = user_input_svd.astype('float32')
    faiss.normalize_L2(user_input_svd)
    D, I = index.search(user_input_svd, k=5)
    similar_recipes = df.iloc[I[0]]
    return similar_recipes[['title', 'link', 'ingredients']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_recipes', methods=['POST'])
def search_recipes():
    data = request.json
    user_ingredients = data.get('ingredients', [])
    recipes = find_similar_recipes(user_ingredients)
    return recipes.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
