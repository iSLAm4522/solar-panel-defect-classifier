import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor

ONNX_MODEL_PATH = Path("onnx_output/model.onnx")
CONFIG_PATH = Path("onnx_output/config.json")
PREPROCESSOR_PATH = "onnx_output/"


@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at '{ONNX_MODEL_PATH}'")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at '{CONFIG_PATH}'")

    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    processor = AutoImageProcessor.from_pretrained(PREPROCESSOR_PATH, use_fast=False)

    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    id2label = {int(k): v for k, v in config_data.get("id2label", {}).items()}
    input_name = session.get_inputs()[0].name

    return session, processor, id2label, input_name


def predict(image: Image.Image, session, processor, id2label, input_name):
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)

    logits = session.run(None, {input_name: pixel_values})[0]

    # Softmax for confidence scores
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    predicted_class_id = int(np.argmax(probabilities, axis=-1)[0])
    confidence = float(probabilities[0][predicted_class_id] * 100)
    class_name = id2label.get(predicted_class_id, f"Class {predicted_class_id}")

    # Full ranked list of all classes
    all_probs = {
        id2label.get(i, f"Class {i}"): float(p * 100)
        for i, p in enumerate(probabilities[0])
    }
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return class_name, confidence, all_probs


def main():
    st.set_page_config(page_title="ViT Solar Panel Defect Classifier", page_icon="🖼️", layout="centered")

    st.title("☀️ Solar Panel Defect Classifier (ViT)")
    st.write("Upload an image of a solar panel to detect potential defects.")

    try:
        session, processor, id2label, input_name = load_resources()
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the uploaded image: {e}")
        return

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            class_name, confidence, all_probs = predict(image, session, processor, id2label, input_name)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    st.success("Prediction complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Defect", value=class_name)
    with col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    with st.expander("Show probabilities for all classes"):
        for label, prob in all_probs.items():
            st.write(f"**{label}**: {prob:.2f}%")
            st.progress(min(prob / 100, 1.0))


if __name__ == "__main__":
    main()
