import pickle
import os
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_single_input

def load_artifacts():
    """
    Loads the trained Keras model and the saved Tokenizer pickle file.
    """
    model_path = 'fake_news_lstm_model.h5'
    tokenizer_path = 'tokenizer.pickle'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
        
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None

def predict_news(text, model, tokenizer):
    """
    Accepts raw text context, preprocesses it, predicts its authenticity,
    and returns a user-friendly result string.
    """
    if not text.strip():
        return "Please enter some news text."
        
    # Preprocess the text into padded sequences
    padded_input = preprocess_single_input(text, tokenizer)
    
    # Predict (Output is a probability between 0 and 1)
    prediction = model.predict(padded_input)[0][0]
    
    # threshold 0.5: >= 0.5 => Real (1), < 0.5 => Fake (0)
    if prediction >= 0.5:
        return f"Real News (Confidence: {prediction*100:.2f}%)"
    else:
        return f"Fake News (Confidence: {(1-prediction)*100:.2f}%)"
