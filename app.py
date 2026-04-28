import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
page_title="Dog Breed Detector",
layout="centered"
)

# Hide sidebar

st.markdown(""" <style>
section[data-testid="stSidebar"] {display: none;} </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
model = models.mobilenet_v2(weights="DEFAULT")
model.eval()
return model

model = load_model()

# -------------------- LOAD LABELS --------------------

@st.cache_resource
def load_labels():
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
return labels

labels = load_labels()

# -------------------- TRANSFORM --------------------

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

# -------------------- UI --------------------

st.title("🐶 AI Dog Breed Detection System")
st.write("Upload a dog image and the model will try to identify the breed.")

uploaded_file = st.file_uploader(
"Choose a dog image...",
type=["jpg", "png", "jpeg"]
)

# -------------------- PREDICTION --------------------

if uploaded_file is not None:
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_column_width=True)

```
img = transform(image).unsqueeze(0)

with st.spinner("Analyzing image..."):
    with torch.no_grad():
        preds = model(img)
        probs = torch.nn.functional.softmax(preds[0], dim=0)

# Top 3 predictions
top_probs, top_idxs = torch.topk(probs, 3)
results = [(labels[i], top_probs[j].item()) for j, i in enumerate(top_idxs)]

# Filter dog-related predictions
dog_keywords = ["dog", "retriever", "shepherd", "terrier", "hound", "spaniel"]
filtered = [r for r in results if any(k in r[0] for k in dog_keywords)]

# Select best result
if filtered:
    top_label = filtered[0][0]
    top_prob = filtered[0][1] * 100
else:
    top_label = results[0][0]
    top_prob = results[0][1] * 100

# Display result
st.success(f"Predicted Breed: {top_label.replace('_',' ').title()}")
st.write(f"Confidence: {top_prob:.2f}%")
```

else:
st.info("Please upload an image to get started.")
