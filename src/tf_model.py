import pandas as pd
import tensorflow as tf
from src.preprocess import clean_text


def load_data(csv_path, max_words=10000, max_len=100, sample_size=2000):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Explicit mapping for your dataset
    df = df[["text", "song", "artist"]]
    df.columns = ["lyrics", "song", "artist"]

    # Remove missing values
    df = df.dropna()

    # 🔥 LIMIT DATASET SIZE FOR FAST & STABLE TRAINING
    # (Important for demo + CPU training)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Clean lyrics
    df["clean_lyrics"] = df["lyrics"].apply(clean_text)

    # Create labels (one class per song in sampled data)
    df["label"] = df.index

    # Tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df["clean_lyrics"])

    sequences = tokenizer.texts_to_sequences(df["clean_lyrics"])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post"
    )

    return padded, df["label"], tokenizer, df


def build_model(vocab_size, max_len, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
