import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="Image Classification with PyTorch", layout="centered")
st.title("🔍 Image Classification App (PyTorch)")

# Load the PyTorch model
@st.cache_resource
def load_model():
    model = torch.load("torch_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Load class labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define preprocessing transformation (resizing and normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to tensor [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Upload an image
uploaded_file = st.file_uploader("📷 Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_index = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_index]

    # Display prediction
    st.success(f"🎯 **Predicted Class:** {predicted_class}")
    st.info(f"📊 **Confidence Score:** {confidence.item() * 100:.2f}%")
