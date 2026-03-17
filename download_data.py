import pandas as pd
import requests
import io
import time

print("Starting download...")
url = "https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip"

try:
    # Use pandas to read directly from a reliable fast source
    print("Reading CSV directly from Kaggle-like dataset on github...")
    
    df = pd.read_csv("https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv")
    
    print("Downloaded shape:", df.shape)
    
    # Needs text and label columns
    df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})
    
    df[['text', 'label']].to_csv('dataset.csv', index=False)
    print("dataset.csv created successfully.")
except Exception as e:
    print("Failed to download via URL 1:", e)
    try:
        url2 = "https://raw.githubusercontent.com/mrvohra/fake-news-detection/master/data/train.csv"
        df2 = pd.read_csv(url2)
        # Usually has title, author, text, label
        # 1: unreliable, 0: reliable (Kaggle Fake News). We want 1: Real, 0: Fake
        # So we flip
        df2['label'] = df2['label'].apply(lambda x: 0 if x == 1 else 1)
        df2[['text', 'label']].dropna().to_csv('dataset.csv', index=False)
        print("dataset.csv created from URL 2.")
    except Exception as e2:
        print("Failed to download via URL 2:", e2)

