# 🎵 Spotify Lyric Search – Machine Learning Project

## 📌 Project Overview
This project implements a **lyric-based song identification system** using the **Spotify 50k+ Songs Dataset**.  
Given a small snippet of song lyrics, the system attempts to identify the **Song Title** and **Artist** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The project demonstrates two approaches:
- A **TensorFlow-based text classification model**
- A **similarity-based lyric search model** using **TF-IDF and cosine similarity**

This comparison highlights real-world challenges in identifying songs from short and ambiguous lyric snippets.

---

## 🎯 Problem Statement
**Input:** A short lyric snippet (text)  
**Output:** Predicted **Song Title** and **Artist**

The goal is to explore how machine learning models perform on lyric identification tasks when limited contextual information is available.

---

## 🛠️ Technologies Used
- **Python 3.11**
- **TensorFlow (Keras)**
- **Scikit-learn**
- **NLTK**
- **Pandas**
- **NumPy**
- **Jupyter Notebook**

---

## 📂 Project Structure

```text
spotify-lyric-search/
│
├── data/
│ └── spotify_lyrics.csv
│
├── src/
│ ├── preprocess.py # Text cleaning & preprocessing
│ ├── tf_model.py # TensorFlow model & data loader
│ ├── similarity_model.py # TF-IDF + cosine similarity model
│ └── predict.py # Main executable script
│
├── notebooks/
│ └── lyric_search_model.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```text
git clone https://github.com/your-username/spotify-lyric-search.git
cd spotify-lyric-search

```

### 2️⃣ Create & Activate Virtual Environment
```text
python -m venv venv
venv\Scripts\activate    # Windows

```

### 3️⃣ Install Required Dependencies
```text
pip install -r requirements.txt

```
### 4️⃣ Download NLTK Resources
```text
python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
exit()

```

## ▶️ Execution
### 🔹 Option 1: Run TensorFlow Classification Model

This script demonstrates a deep learning-based approach using TensorFlow.
python src/predict.py

The script will:
- Load and preprocess the dataset
- Train a TensorFlow text classification model
- Prompt the user for a lyric snippet
- Output a predicted Song Title and Artist

### 🔹 Option 2: Run Jupyter Notebook (Recommended)

The notebook provides a step-by-step demonstration of both models.
jupyter notebook

Open:
```text
notebooks/lyric_search_model.ipynb
```

The notebook includes:
Dataset preview
Text preprocessing
TensorFlow model training & prediction
TF-IDF similarity-based lyric search
Model comparison and explanation

---

## 🧠 Model Explanation
### 1️⃣ TensorFlow Classification Model

- Tokenization and padding of lyric text
- Embedding layer followed by dense neural layers
- Multi-class classification where each song is treated as a class
- This model satisfies the requirement of using TensorFlow for NLP tasks.

### 2️⃣ Similarity-Based Lyric Search Model

- Lyrics converted into TF-IDF vectors
- Cosine similarity used to retrieve the most similar song lyrics
- This approach performs better for short lyric snippets and reflects common information- retrieval techniques.

## ⚠️ Prediction Behavior & Limitations

- Short lyric snippets often appear in multiple songs
- The TensorFlow classification model may not always return the original source song
- The similarity model retrieves the most statistically similar lyrics, which may differ from -the original song
- These behaviors are expected and represent real-world NLP challenges.

---

## 📊 Dataset

Spotify 50k+ Songs Dataset
Link - 'https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places'

The dataset is not included in this repository due to size constraints.
Please download it separately and place it inside the data/ directory as:

```text
data/spotify_lyrics.csv
```

---

##  Project Requirements Fulfilled

- Python-based implementation
- Uses TensorFlow
- Text preprocessing (tokenization, stop-word removal)
- Classification and similarity models implemented
- Runnable Python script and Jupyter Notebook
- Installation and execution instructions included
- Clean GitHub repository structure

---

## 👨‍💻 Author

**Subham Maity**

---

## 🏁 Final Notes

This project is intended for academic evaluation and demonstrates both deep learning and similarity-based approaches to lyric identification. The focus is on clarity, reproducibility, and practical understanding rather than perfect prediction accuracy.
