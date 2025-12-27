import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess import clean_text


def build_similarity_model(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["text", "song", "artist"]].dropna()
    df["clean_lyrics"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["clean_lyrics"])

    return df, vectorizer, tfidf_matrix


def predict_song_similarity(snippet, df, vectorizer, tfidf_matrix):
    snippet = clean_text(snippet)
    vec = vectorizer.transform([snippet])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()

    idx = scores.argmax()
    return df.iloc[idx]["song"], df.iloc[idx]["artist"], scores[idx]
