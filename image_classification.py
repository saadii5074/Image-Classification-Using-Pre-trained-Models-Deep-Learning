import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("📷 Image Classification using MobileNetV2")
st.write("Upload an image, and the pre-trained MobileNetV2 model will predict what's in it.")

# Load the MobileNetV2 model
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    results = decode_predictions(preds, top=3)[0]

    st.subheader("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(results):
        st.write(f"**{label}**: {score*100:.2f}%")

