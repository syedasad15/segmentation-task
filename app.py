import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Segmentation System",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
}

.sub-title {
    font-size: 16px;
    color: #94a3b8;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 0px 15px rgba(56,189,248,0.2);
}

.metric-box {
    background: #0b1220;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🧠 Brain Tumor Segmentation AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning-based MRI analysis system for tumor localization</div>', unsafe_allow_html=True)

st.write("")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

# ---------------- UPLOAD ----------------
with col1:
    st.markdown("### 📤 Upload MRI Scan")
    file = st.file_uploader("Drop MRI image here", type=["png", "jpg", "jpeg"])

# ---------------- FUNCTIONS ----------------
def preprocess(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def predict(img):
    pred = session.run(None, {input_name: img})[0]
    mask = pred[0][0]
    mask = (mask > 0.5).astype(np.uint8)
    return mask

# ---------------- PROCESS ----------------
if file:

    image = Image.open(file)

    # Original image
    orig = np.array(image.resize((256, 256)))

    # Prediction
    inp = preprocess(image)
    mask = predict(inp)
    mask = cv2.resize(mask, (256, 256))

    # Overlay (IMPORTANT for segmentation visualization)
    overlay = orig.copy()
    overlay[mask == 1] = [255, 0, 0]

    # ---------------- RIGHT SIDE UI ----------------
    with col2:

        st.markdown("### 🖼 Results Dashboard")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Original MRI")
            st.image(orig, use_container_width=True)

        with c2:
            st.markdown("#### Tumor Mask")
            st.image(mask * 255, use_container_width=True)

        with c3:
            st.markdown("#### Overlay Result")
            st.image(overlay, use_container_width=True)

        # ---------------- METRICS ----------------
        tumor_ratio = float(mask.mean() * 100)

        st.write("")
        st.markdown("### 📊 Analysis Report")

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>Tumor Area</h3>
                <h2>{tumor_ratio:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            status = "⚠️ Tumor Detected" if tumor_ratio > 1 else "✅ Normal Region"

            st.markdown(f"""
            <div class="metric-box">
                <h3>Status</h3>
                <h2>{status}</h2>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- EXPLANATION ----------------
        st.info(
            "This system uses a deep learning segmentation model to highlight abnormal brain regions "
            "in MRI scans. Red overlay indicates predicted tumor region."
        )
