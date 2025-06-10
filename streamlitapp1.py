import keras2onnx
from tensorflow.keras.models import load_model
import onnx
from onnx2pytorch import ConvertModel
import torch

# Load Keras model
keras_model = load_model("keras_model.h5")

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
onnx.save_model(onnx_model, "model.onnx")

# Load ONNX and convert to PyTorch
onnx_model = onnx.load("model.onnx")
pytorch_model = ConvertModel(onnx_model)

# Save PyTorch model
torch.save(pytorch_model, "torch_model.pt")
print("✅ Converted to torch_model.pt successfully!")

import keras2onnx
from tensorflow.keras.models import load_model
import onnx
from onnx2pytorch import ConvertModel
import torch

# Load Keras model
keras_model = load_model("keras_model.h5")

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
onnx.save_model(onnx_model, "model.onnx")

# Load ONNX and convert to PyTorch
onnx_model = onnx.load("model.onnx")
pytorch_model = ConvertModel(onnx_model)

# Save PyTorch model
torch.save(pytorch_model, "torch_model.pt")
print("✅ Converted to torch_model.pt successfully!")
