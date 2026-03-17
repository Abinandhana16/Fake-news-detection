import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from data_preprocessing import load_and_preprocess_data, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH

def build_model(vocab_size, max_length):
    """
    Builds the LSTM-based deep learning model structure.
    """
    model = Sequential()
    
    # Embedding layer to convert word indices to dense vectors
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
    
    # LSTM layer to capture sequential/contextual relationships
    model.add(LSTM(units=128, return_sequences=False))
    
    # Dropout layer randomly drops neurons to prevent overfitting
    model.add(Dropout(0.2))
    
    # Dense output layer with sigmoid for binary classification output (0.0 to 1.0)
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_and_evaluate(csv_path):
    """
    Main pipeline for training and evaluating the Fake News LSTM model.
    """
    # 1. Prepare data
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data(csv_path)
    
    # 2. Build model
    print("Building model...")
    model = build_model(MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    model.summary() # Print the architecture
    
    # 3. Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,  # 10% of training data used for validation
        epochs=1,              # Number of times to iterate over the entire dataset
        batch_size=64
    )
    
    # 4. Evaluate
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # 5. Save Model
    print("Saving model...")
    model.save('fake_news_lstm_model.h5')
    print("Model saved to 'fake_news_lstm_model.h5'")

if __name__ == "__main__":
    print("This script trains the Fake News model.")
    # Uncomment the lines below and place a valid dataset to train the model end-to-end
    print("Specify your dataset path: ensure it has 'text' and 'label' columns.")
    dataset_path = "dataset.csv" 
    train_and_evaluate(dataset_path)
