import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Page title
st.title("🖼️ Image Classifier (Keras only - No TensorFlow)")
st.write("Upload an image and classify it using your Keras `.h5` model.")

# File uploader for image input
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare image for model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
# Assuming these constants are defined similarly in your original code
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS = 1 # Anomaly class label

# Load model and labels
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

# Import utility functions from train_evaluate.py
from train_evaluate import get_bbox_from_heatmap # Only need this for prediction


# --- Utility Functions (Adapted for Streamlit) ---

@st.cache_resource
def load_model(model_path="anomaly_detection_model.pth"):
    """
    Loads the pre-trained anomaly detection model.
    Uses st.cache_resource to avoid reloading the model on every rerun.
    """
    try:
        model = CustomVGG(n_classes=2)
        # Check if running on CPU or GPU and load accordingly
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
            model.to("cuda")
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.to("cpu")
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_and_localize_streamlit(model, image, thres=0.8):
    """
    Performs prediction and localization for a single image, adapted for Streamlit.
    """
    class_names = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]
    img_transform = transforms.Compose(
        [transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()]
    )

    # Prepare image
    img_tensor = img_transform(image).unsqueeze(0) # Add batch dimension

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        out = model(img_tensor)
        probs, feature_maps = out[0], out[1]

    preds_probs = torch.softmax(probs, dim=-1) # Ensure probabilities are calculated correctly
    preds_class = torch.argmax(preds_probs, dim=-1)
    
    # Get values for the single image
    class_pred = preds_class.item()
    prob = preds_probs[0, class_pred].item() # Probability of the predicted class
    
    # Get heatmap for the anomaly class
    heatmap = feature_maps[0, NEG_CLASS].cpu().numpy()

    # Create a matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")

    # Set title
    title_text = f"Predicted: {class_names[class_pred]}, Prob: {prob:.3f}"
    ax.set_title(title_text)

    # If anomaly is predicted, draw bounding box and optionally heatmap
    if class_pred == NEG_CLASS:
        x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)
        rectangle = Rectangle(
            (x_0, y_0),
            x_1 - x_0,
            y_1 - y_0,
            edgecolor="red",
            facecolor="none",
            lw=3,
        )
        ax.add_patch(rectangle)
        
        # Overlay heatmap
        ax.imshow(heatmap, cmap="Reds", alpha=0.3)

    return fig, class_names[class_pred], prob


# --- Streamlit UI ---

st.set_page_config(page_title="Anomaly Detection App", layout="centered")

st.title("🏭 Anomaly Detection for Industrial Inspection")

st.write(
    "Upload an image (e.g., from an industrial product) to detect anomalies. "
    "The model will classify the image as 'Good' or 'Anomaly' and, "
    "if an anomaly is detected, highlight the defective region."
)

# Sidebar for model loading (if you had multiple models or options)
st.sidebar.header("Model Configuration")
model_file = st.sidebar.file_uploader(
    "Upload your model file (.pth)", type=["pth"]
)

model = None
if model_file:
    # Save the uploaded model file temporarily to load it
    with open("uploaded_model.pth", "wb") as f:
        f.write(model_file.getbuffer())
    model = load_model("uploaded_model.pth")
    st.sidebar.success("Model loaded successfully!")
else:
    st.sidebar.info("Please upload a `.pth` model file to proceed.")


# Main upload section
st.header("Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting anomaly...")

    # Anomaly detection threshold slider
    anomaly_threshold = st.slider(
        "Set Anomaly Detection Threshold (for bounding box)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Higher threshold means only very strong anomaly signals will trigger a bounding box."
    )

    # Perform prediction and localization
    fig, predicted_class, probability = predict_and_localize_streamlit(model, image, anomaly_threshold)

    st.subheader("Prediction Result:")
    if predicted_class == ("Good" if NEG_CLASS == 0 else "Anomaly"): # Check against the actual anomaly class
        st.error(f"**Anomaly Detected!** (Confidence: {probability:.2f})")
    else:
        st.success(f"**Image is Good!** (Confidence: {probability:.2f})")

    st.pyplot(fig) # Display the plot with bounding box/heatmap

elif uploaded_file is not None and model is None:
    st.warning("Please upload a trained model file first to perform predictions.")
else:
    st.info("Upload an image and a model file to get started!")
