import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Page config

st.set_page_config(page_title="Dog Breed Detector", layout="centered")

# Load pretrained model

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Image preprocessing

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
])

# UI

st.title("🐶 AI Dog Breed Detection System")
st.write("Upload an image and the model will predict.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

img = transform(image).unsqueeze(0)

with st.spinner("Analyzing image..."):
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

st.success(f"Prediction Index: {predicted.item()}")


else:
st.info("Please upload an image to start.")
