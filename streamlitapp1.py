import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Page title
st.title("🖼️ Image Classifier (Keras only - No TensorFlow)")
st.write("Upload an image and classify it using your Keras `.h5` model.")

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Resize and normalize
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare input for model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Show result
    st.markdown("### 🎯 Prediction")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
else:
    st.info("Please upload an image to classify.")
