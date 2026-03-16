import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
import matplotlib.cm as cm
import os
import zipfile
import json
import tempfile

st.set_page_config(page_title="Deepfake Detector XAI", layout="wide")
st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)

def get_model_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base_path, "MobileNetV2_best.keras"),
              os.path.join(base_path, "models", "MobileNetV2_best.keras")]:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_forensic_model():
    model_path = get_model_path()
    if not model_path:
        st.error("Model file not found.")
        return None

    try:
        # Build model with exact architecture: MobileNetV2 -> GAP -> Dense(256,relu) -> Dense(1,sigmoid)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)

        # Extract and load weights from the .keras zip file
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)

            weights_path = None
            for fname in os.listdir(tmpdir):
                if fname.endswith('.h5') or 'weight' in fname.lower():
                    weights_path = os.path.join(tmpdir, fname)
                    break

            if weights_path:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            else:
                st.warning(f"No weights file found in zip.")

        return model

    except Exception as e:
        st.error(f"Model build failed: {e}")
        return None


def get_gradcam_heatmap(img_array, model):
    try:
        base = model.layers[1]  # MobileNetV2 base
        last_conv = base.get_layer("out_relu")
        grad_model = tf.keras.Model(
            inputs=base.input,
            outputs=[last_conv.output, base.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None


model = load_forensic_model()

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(temp_path)

    if st.button("Generate Forensic Report", type="primary"):
        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Analyzing..."):
                cap = cv2.VideoCapture(temp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    st.error("Frame extraction failed.")
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_array = np.expand_dims(img_resized, axis=0).astype("float32") / 255.0

                    preds = model.predict(img_array, verbose=0)
                    prediction = float(preds[0][0])

                    label = "AI (Deepfake)" if prediction < 0.5 else "Real Video"
                    conf = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Verdict", label)
                    c2.metric("Confidence", f"{conf:.2f}%")

                    heatmap = get_gradcam_heatmap(img_array, model)
                    if heatmap is not None:
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_uint8 = np.uint8(255 * heatmap_resized)
                        jet_heatmap = matplotlib.colormaps["jet"](np.arange(256))[:, :3][heatmap_uint8]
                        super_img = np.clip(jet_heatmap * 0.4 + img_resized / 255.0, 0, 1)
                        col1, col2 = st.columns(2)
                        col1.image(img_resized, caption="Extracted Frame")
                        col2.image(super_img, caption="Grad-CAM XAI Map")
                    else:
                        st.image(img_resized, caption="Extracted Frame (Grad-CAM unavailable)")
