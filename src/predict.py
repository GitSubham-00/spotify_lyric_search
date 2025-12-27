import tensorflow as tf
from tf_model import load_data, build_model
from preprocess import clean_text

DATA_PATH = "data/spotify_lyrics.csv"
MAX_WORDS = 10000
MAX_LEN = 100

print("🔄 Loading data...")
X, y, tokenizer, df = load_data(DATA_PATH, MAX_WORDS, MAX_LEN)

print("🧠 Building model...")
model = build_model(MAX_WORDS, MAX_LEN, len(df))

print("⚠️ Training model (quick demo training)...")
model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)

def predict_song(snippet):
    snippet = clean_text(snippet)
    seq = tokenizer.texts_to_sequences([snippet])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)
    index = prediction.argmax()

    return df.iloc[index]["song"], df.iloc[index]["artist"]

if __name__ == "__main__":
    text = input("\nEnter lyric snippet: ")
    song, artist = predict_song(text)
    print(f"\n🎵 Song: {song}")
    print(f"🎤 Artist: {artist}")
