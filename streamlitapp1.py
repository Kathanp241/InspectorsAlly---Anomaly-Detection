import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Data Download (Crucial for deployment) ---
# This block ensures that necessary NLTK data is available.
# Streamlit Cloud often handles 'stopwords' and 'punkt' automatically if called
# in the app, but it's good practice to ensure they are downloaded.
try:
    # Check if stopwords are already downloaded
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    # If not, download them. This might take a moment on first run/deployment.
    nltk.download('stopwords')
try:
    # Check if punkt tokenizer is already downloaded (often used by other NLTK functions)
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Load Trained Model and TF-IDF Vectorizer ---
# The model and vectorizer are saved as .pkl files after training.
# We load them here to use for predictions in the Streamlit app.
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    # Display an error message if the model files are not found.
    # This is important for debugging during local development or deployment.
    st.error("Error: 'model.pkl' or 'tfidf_vectorizer.pkl' not found. "
             "Please ensure you have run 'train_classifier.py' to train and save the model.")
    # Stop the Streamlit app execution if essential files are missing
    st.stop()

# --- Text Preprocessing Function ---
# This function must be IDENTICAL to the one used in `train_classifier.py`
# to ensure consistent feature engineering.
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps include: lowercasing, removing non-alphanumeric characters,
    tokenization, removing stopwords, and stemming.
    """
    text = text.lower() # Convert text to lowercase
    # Remove punctuation and special characters, keep only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split() # Split text into individual words (tokenization)
    # Remove common English stopwords and apply stemming (reducing words to their root form)
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words) # Join the processed words back into a single string

# --- Streamlit App Interface ---
# Set basic page configuration for the web app.
st.set_page_config(
    page_title="Spam Email Classifier", # Title that appears in the browser tab
    page_icon="📧", # Emoji icon for the browser tab
    layout="centered" # Page layout (can be "wide" or "centered")
)

# Main title of the application
st.title("📧 Spam Email Classifier")
st.markdown("---") # Horizontal line for visual separation

# Introductory text for the user
st.markdown(
    """
    Enter an email or SMS message in the text area below to check if it's spam or not.
    Our AI model will analyze the content and tell you its prediction!
    """
)

# Text area for user input
user_input = st.text_area(
    "Your Message:",
    height=200, # Height of the text area
    placeholder="Type your email or SMS content here...", # Placeholder text
    help="Enter the full content of the message you want to classify." # Tooltip help text
)

# Button to trigger classification
if st.button("Classify Message"):
    if user_input: # Check if user has entered any text
        with st.spinner("Classifying..."): # Show a spinner while processing
            # 1. Preprocess the user's input text
            processed_input = preprocess_text(user_input)

            # 2. Transform the preprocessed text into numerical features
            # using the same TF-IDF vectorizer that was trained with the model.
            # .transform expects an iterable, so we pass [processed_input].
            vectorized_input = tfidf_vectorizer.transform([processed_input])

            # 3. Make a prediction using the loaded machine learning model.
            prediction = model.predict(vectorized_input)
            # Get the probability scores for each class (spam and ham)
            prediction_proba = model.predict_proba(vectorized_input)

        # Display the result to the user based on the prediction
        if prediction[0] == 1:
            st.warning(
                f"🚨 **This is likely SPAM!** "
                f"(Confidence: {prediction_proba[0][1]*100:.2f}%)"
            )
            st.write("Spam messages often contain suspicious links, unusual requests, or unsolicited offers.")
        else:
            st.success(
                f"✅ **This is NOT Spam (HAM).** "
                f"(Confidence: {prediction_proba[0][0]*100:.2f}%)"
            )
            st.write("This message appears to be legitimate.")
    else:
        # Inform the user if no text was entered
        st.info("Please enter some text in the box above to classify.")

st.markdown("---") # Another horizontal line

# --- Custom CSS for Styling ---
# This section adds some basic styling to make the app look better.
# Using `unsafe_allow_html=True` is necessary for injecting custom CSS.
st.markdown(
    """
    <style>
        /* Style for the text area */
        .stTextArea [data-baseweb="textarea"] {
            background-color: #ffffff; /* White background */
            border: 1px solid #d3d3d3; /* Light grey border */
            border-radius: 8px; /* Rounded corners */
            padding: 15px; /* Inner spacing */
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow */
        }

        /* Style for the classify button */
        .stButton > button {
            background-color: #4CAF50; /* Green background */
            color: white; /* White text */
            font-weight: bold; /* Bold text */
            padding: 12px 25px; /* Padding for the button */
            border-radius: 10px; /* More rounded corners */
            border: none; /* No default border */
            cursor: pointer; /* Pointer cursor on hover */
            transition: background-color 0.3s ease; /* Smooth transition for hover effect */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Shadow for button */
        }

        /* Hover effect for the button */
        .stButton > button:hover {
            background-color: #45a049; /* Slightly darker green on hover */
            box-shadow: 0 6px 8px rgba(0,0,0,0.15); /* Slightly larger shadow on hover */
        }

        /* General font styling */
        body {
            font-family: 'Inter', sans-serif; /* Use Inter font */
        }
    </style>
    """,
    unsafe_allow_html=True # Allow Streamlit to render custom HTML/CSS
)
