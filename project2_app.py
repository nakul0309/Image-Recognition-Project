"""
Project 2 — Image Recognition (Streamlit App)
================================================
Run locally with:
    streamlit run app.py

Deployed on Streamlit Community Cloud by pointing it at this file
in your GitHub repo.
"""

import time
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Image Recognition", page_icon="🖼️")
st.title("🖼️ Image Recognition")
st.write(
    "This app uses **MobileNetV2** (pretrained on ImageNet, 1000 object "
    "categories) to identify what's in a photo. Upload any image to try it."
)


# ------------------------------------------------------------------
# Load the pretrained model once and cache it across users/interactions
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with st.spinner("Loading pretrained model (first run only)..."):
        model = MobileNetV2(weights="imagenet")
    return model


image_model = load_model()

with st.sidebar:
    st.header("Model Info")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Trained on:** ImageNet (1000 categories)")
    st.caption(
        "This model was never shown your specific photo during training — "
        "it's applying patterns learned from 1M+ other images."
    )

# ------------------------------------------------------------------
# Image upload
# ------------------------------------------------------------------
st.subheader("Upload an image")
uploaded_file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    # Resize to what MobileNetV2 expects
    resized_image = image.resize((224, 224))

    # Prepare the image: array -> batch -> preprocess
    image_array = img_to_array(resized_image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)

    # Predict
    with st.spinner("Analyzing image..."):
        start_time = time.time()
        predictions = image_model.predict(processed_image, verbose=0)
        elapsed_ms = (time.time() - start_time) * 1000

    decoded_predictions = decode_predictions(predictions, top=5)[0]

    st.subheader("Top Predictions")
    st.caption(f"Prediction took {elapsed_ms:.0f} ms")

    for _, label, probability in decoded_predictions:
        label_display = label.replace("_", " ").title()
        st.write(f"**{label_display}**")
        st.progress(float(probability))
        st.caption(f"{probability * 100:.2f}% confidence")
else:
    st.info("👆 Upload an image above to get a prediction.")

st.divider()
st.caption(
    "Built with TensorFlow (MobileNetV2, transfer learning) and Streamlit. "
    "High confidence reflects strong pattern match, not guaranteed correctness."
)
