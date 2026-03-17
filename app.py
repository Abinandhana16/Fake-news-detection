import streamlit as st
import os
from prediction import load_artifacts, predict_news

# Set up the visual look of the Streamlit page
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection using LSTM")
st.markdown("### Identify misinformation using deep learning!")
st.write("Enter a news article text below. Our pre-trained LSTM neural network will analyze linguistic patterns to detect whether it's **Real** or **Fake**.")

# Check for model existence before attempting prediction
model, tokenizer = load_artifacts()

if model is None or tokenizer is None:
    st.warning("⚠️ **Model or Tokenizer not found!**")
    st.info("You need to train your model first. Place your `dataset.csv` in the project root and run `python model_training.py`.")
else:
    # Text Area for user input
    user_text = st.text_area("✍️ News Text", height=200, placeholder="Paste your news article here...")
    
    # Trigger prediction on button click
    if st.button("🔍 Check News"):
        if user_text:
            with st.spinner("Analyzing text..."):
                result = predict_news(user_text, model, tokenizer)
                
                # Display outcome prominently
                st.markdown("---")
                if "Real News" in result:
                    st.success(f"✅ **{result}**")
                    st.balloons()
                elif "Fake News" in result:
                    st.error(f"🚨 **{result}** - Be careful!")
                else:
                    st.warning(result)
        else:
            st.warning("Please provide some text to analyze.")
