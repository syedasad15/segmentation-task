import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Segmentation AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # IMPORTANT DEBUG

# ---------------- UI HEADER ----------------
st.title("🧠 Brain Tumor Segmentation AI")
st.markdown("Upload MRI scan and get AI-based tumor segmentation result")

# ---------------- SIDEBAR DEBUG (VERY USEFUL) ----------------
with st.sidebar:
    st.header("🔧 Model Info")
    st.write("Input Name:", input_name)
    st.write("Input Shape:", input_shape)

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

# ---------------- PREPROCESS FUNCTION (FIXED) ----------------
def preprocess(image):
    image = np.array(image)

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get model size safely
    try:
        _, _, H, W = input_shape
        H = H if isinstance(H, int) else 256
        W = W if isinstance(W, int) else 256
    except:
        H, W = 256, 256

    image = cv2.resize(image, (W, H))

    image = image.astype(np.float32) / 255.0

    # CHW format
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image

# ---------------- PREDICT FUNCTION ----------------
def predict(image):
    pred = session.run(None, {input_name: image})[0]

    # Handle common segmentation outputs
    if len(pred.shape) == 4:
        pred = pred[0]

    if pred.shape[0] == 1:
        mask = pred[0]
    else:
        mask = np.argmax(pred, axis=0)

    mask = (mask > 0.5).astype(np.uint8)

    return mask

# ---------------- MAIN APP ----------------
if file:
    image = Image.open(file).convert("RGB")

    col1, col2, col3 = st.columns(3)

    # ORIGINAL IMAGE
    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    # PREPROCESS
    input_tensor = preprocess(image)

    # PREDICT
    try:
        mask = predict(input_tensor)

        # Resize back to display
        orig = np.array(image.resize((256, 256)))
        mask_resized = cv2.resize(mask, (256, 256))

        overlay = orig.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        # MASK
        with col2:
            st.subheader("Tumor Mask")
            st.image(mask_resized * 255, use_container_width=True)

        # OVERLAY
        with col3:
            st.subheader("Overlay Result")
            st.image(overlay, use_container_width=True)

        # METRICS
        tumor_ratio = float(mask_resized.mean() * 100)

        st.markdown("### 📊 Analysis Report")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Tumor Area %", f"{tumor_ratio:.2f}%")

        with c2:
            status = "⚠️ Tumor Detected" if tumor_ratio > 1 else "✅ No Significant Tumor"
            st.metric("Status", status)

        st.success("Segmentation completed successfully")

    except Exception as e:
        st.error("Model inference failed")
        st.exception(e)

