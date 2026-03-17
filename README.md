# Fake News Detection using LSTM

This project is a complete deep-learning workflow designed to detect and classify news articles as "Real" or "Fake" using Long Short-Term Memory (LSTM) recurrent neural networks. It features data preprocessing pipelines, deep learning model building, prediction inference scripts, and a Streamlit-based graphical user interface.

## 📂 Project Structure

- `data_preprocessing.py`: Contains functions to clean text (lowercase, punctuation removal), tokenize datasets using Keras Tokenizer, convert words into padded sequences, and split data into training and testing sets.
- `model_training.py`: Constructs the LSTM architecture containing an `Embedding` layer, `LSTM` layers, and a `Dense` output layer with binary crossentropy, and compiles/trains it. 
- `prediction.py`: Loads the trained Keras model (`.h5`) and Tokenizer configuration (`.pickle`) to seamlessly predict probabilities on real-time raw news text.
- `app.py`: A simple and beginner-friendly Streamlit frontend providing text input capabilities and results visualization.
- `requirements.txt`: Requirements necessary to run the project.

## 💾 Dataset Format

To train your application, you must supply a dataset containing at least two columns:
1. `text`: The raw text of the news article.
2. `label`: Binary classification where `1` = Real, and `0` = Fake.

**Recommended publicly available datasets:**
* [ISOT Fake News Dataset (Kaggle)](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)
* [Fake and real news dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

*Combine the fake and real news into one CSV and assign 1/0 labels before training!*

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. From the project directory, run:
```bash
pip install -r requirements.txt
```

### 2. Train the LSTM model
1. Download a dataset and name it `dataset.csv` (or edit `model_training.py` with your path).
2. Uncomment the training lines in the `if __name__ == "__main__":` block inside `model_training.py`.
3. Train by executing:
```bash
python model_training.py
```
This produces the `fake_news_lstm_model.h5` and `tokenizer.pickle` artifacts. 

### 3. Launch the Frontend Application
Once trained, to use the interactive interface, start the Streamlit server:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.
