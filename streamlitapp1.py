from keras.models import load_model  # type: ignore # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # type: ignore # Install pillow instead of PIL
import numpy as np # type: ignore
import streamlit as st  # type: ignore

# --- CSS Styling for Background and Fonts ---
# --- Dark Theme Custom CSS Styling ---
st.markdown(
    """
    <style>
        /* Main background and text colors */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #f5f5f5;
        }

        /* Fonts and padding */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
        }

        /* File uploader and other widgets */
        .stFileUploader, .stButton>button {
            background-color: #1f4068;
            color: #ffffff;
            border: none;
            padding: 0.5em 1em;
            border-radius: 8px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #3c6382;
            color: #ffffff;
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #00cec9;
        }

        .stImage {
            border: 2px solid #00cec9;
            border-radius: 10px;
        }

        .prediction-good {
            color: #00ffab;
            font-weight: bold;
            font-size: 18px;
        }

        .prediction-bad {
            color: #ff7675;
            font-weight: bold;
            font-size: 18px;
        }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 200, 3), dtype=np.float32)



# Upload image
uploaded_image = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open and display the image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    size = (224, 200)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized).astype(np.float32)
    normalized_image = (image_array / 127.5) - 1

    # Prepare for prediction
    data = np.ndarray((1, 224, 200, 3), dtype=np.float32)
    data[0] = normalized_image

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    # Display prediction result
    st.markdown("## 🧠 Prediction Result")
    # Remove possible label index or prefix (e.g., "0 Good" -> "Good")
    clean_class_name = class_name.split()[-1].lower()
    if clean_class_name == "good":
        st.markdown(
            f"<div class='prediction-good'>✅ Class: {class_name}<br>🔒 Confidence: {confidence_score:.2f}</div>",
            unsafe_allow_html=True,
        )
        st.success(
            "🎉 Congratulations! Your product has been classified as a **Good** item. No anomalies detected."
        )
    else:
        st.markdown(
            f"<div class='prediction-bad'>⚠️ Class: {class_name}<br>🚨 Confidence: {confidence_score:.2f}</div>",
            unsafe_allow_html=True,
        )
        st.error(
            "⚠️ Our AI-based inspection has detected an **Anomaly** in your product."
        )
