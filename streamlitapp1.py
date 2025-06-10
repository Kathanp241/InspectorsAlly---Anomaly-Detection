import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from keras.models import load_model as keras_load_model  # Keras only
import os

# --- Page Configuration ---
st.set_page_config(page_title="Industrial Anomaly & Image Classification", layout="centered")

# --- Constants ---
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS = 1  # "Anomaly" class index

# --- Titles and Descriptions ---
st.title("🏭 Industrial Inspection App")
st.write("""
This tool performs two tasks:
1. Detect anomalies in industrial images using a PyTorch model.
2. Classify images using a Keras `.h5` model.
Upload your image and choose the model to run the appropriate task.
""")

# --- Sidebar: Load Models ---
st.sidebar.header("Model Upload")
torch_model_file = st.sidebar.file_uploader("Upload PyTorch Model (.pth)", type=["pth"])
keras_model_file = st.sidebar.file_uploader("Upload Keras Model (.h5)", type=["h5"])

# --- Utility: Load PyTorch Model ---
@st.cache_resource
def load_torch_model(path):
    try:
        model = CustomVGG(n_classes=2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None

# --- Utility: Load Keras Model and Labels ---
@st.cache_resource
def load_keras_model():
    model = keras_load_model("keras_Model.h5", compile=False)
    labels = open("labels.txt").read().splitlines()
    return model, labels

# --- Utility: Predict & Localize Anomaly (PyTorch) ---
def predict_and_localize(model, image, threshold=0.8):
    transform = transforms.Compose([transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output, features = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs).item()
        prob = probs[0][pred_class].item()
        heatmap = features[0][NEG_CLASS].cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"Prediction: {'Anomaly' if pred_class == NEG_CLASS else 'Good'} ({prob:.2f})")

    if pred_class == NEG_CLASS:
        x0, y0, x1, y1 = get_bbox_from_heatmap(heatmap, threshold)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', lw=3)
        ax.add_patch(rect)
        ax.imshow(heatmap, cmap='Reds', alpha=0.3)

    return fig, pred_class, prob

# --- Main: Image Upload ---
st.header("📤 Upload Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- Option 1: Anomaly Detection ---
    if torch_model_file:
        with open("uploaded_model.pth", "wb") as f:
            f.write(torch_model_file.getbuffer())
        torch_model = load_torch_model("uploaded_model.pth")
        threshold = st.slider("Anomaly Detection Threshold", 0.0, 1.0, 0.8, 0.05)
        st.write("🔍 Performing anomaly detection...")
        fig, pred_class, prob = predict_and_localize(torch_model, image, threshold)
        st.pyplot(fig)
        if pred_class == NEG_CLASS:
            st.error(f"🚨 Anomaly Detected! Confidence: {prob:.2f}")
        else:
            st.success(f"✅ No Anomaly. Confidence: {prob:.2f}")

    # --- Option 2: Keras Image Classification ---
    elif keras_model_file:
        with open("keras_Model.h5", "wb") as f:
            f.write(keras_model_file.getbuffer())

        keras_model, class_names = load_keras_model()

        # Preprocess image
        resized = ImageOps.fit(image, INPUT_IMG_SIZE, Image.Resampling.LANCZOS)
        array = np.asarray(resized).astype(np.float32)
        normalized = (array / 127.5) - 1
        data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized

        # Predict
        predictions = keras_model.predict(data)
        index = np.argmax(predictions)
        confidence = predictions[0][index]
        label = class_names[index]

        st.markdown("### 🧠 Keras Classification Result")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.info("Upload either a PyTorch or Keras model to get started.")
else:
    st.info("Please upload an image first.")

