import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Hyperparameters for text processing
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 500

def clean_text(text):
    """
    Cleans the input text by converting to lowercase and removing punctuation.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    # Remove bracketed text, punctuation, URLs, HTML tags, newlines, and digits
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def load_and_preprocess_data(csv_path):
    """
    Loads dataset, cleans text, tokenizes, pads sequences, and splits into train/test.
    Assumes dataset has 'text' and 'label' columns, where 1=Real and 0=Fake.
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Apply text cleaning
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Tokenization: Convert words to word indices
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df['clean_text'])
    
    # Convert text strings into sequences of numeric indices
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    
    # Padding sequences to ensure equal input length for LSTM
    print("Padding sequences...")
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = df['label'].values
    
    # Split the dataset into 80% training and 20% testing
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the fitted tokenizer so we can use it on new text during inference
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return X_train, X_test, y_train, y_test, tokenizer

def preprocess_single_input(text, tokenizer):
    """
    Preprocess a single text input string for the prediction pipeline.
    """
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    return padded
