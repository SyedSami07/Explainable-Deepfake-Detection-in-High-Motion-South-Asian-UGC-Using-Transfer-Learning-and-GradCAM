import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
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

def inspect_keras_zip(model_path):
    """Peek inside the .keras zip to find Dense layer shapes."""
    dense_units = []
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            if 'config.json' in zf.namelist():
                config = json.loads(zf.read('config.json'))
                config_str = json.dumps(config)
                # Find all Dense layer unit counts from config
                def find_dense(obj):
                    if isinstance(obj, dict):
                        if obj.get('class_name') == 'Dense':
                            units = obj.get('config', {}).get('units')
                            if units:
                                dense_units.append(units)
                        for v in obj.values():
                            find_dense(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_dense(item)
                find_dense(config)
    except Exception as e:
        st.warning(f"Could not inspect config: {e}")
    return dense_units

@st.cache_resource
def load_forensic_model():
    model_path = get_model_path()
    if not model_path:
        st.error("Model file not found.")
        return None

    # Inspect what Dense layers exist in the saved model
    dense_units = inspect_keras_zip(model_path)
    st.info(f"Detected Dense layers in saved model: {dense_units}")

    try:
        # Build MobileNetV2 backbone
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=None
        )

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Dynamically add Dense layers matching the saved model
        # Last unit is always the output (sigmoid), rest are hidden (relu)
        if len(dense_units) >= 2:
            for units in dense_units[:-1]:
                x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.Dense(dense_units[-1], activation='sigmoid')(x)
        elif len(dense_units) == 1:
            x = tf.keras.layers.Dense(dense_units[0], activation='sigmoid')(x)
        else:
            # Fallback: assume Dense(256) -> Dense(1) based on shape error
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, x)

        # Load weights from inside the .keras zip
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)

            weights_path = None
            for fname in os.listdir(tmpdir):
                if fname.endswith('.h5') or 'weight' in fname.lower():
                    weights_path = os.path.join(tmpdir, fname)
                    break

            if weights_path and os.path.exists(weights_path):
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                st.success("Weights loaded successfully.")
            else:
                files = os.listdir(tmpdir)
                st.warning(f"No weights file found. Contents: {files}")

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
                        # Fixed: use colormaps instead of deprecated get_cmap
                        jet_heatmap = matplotlib.colormaps["jet"](np.arange(256))[:, :3][heatmap_uint8]
                        super_img = np.clip(jet_heatmap * 0.4 + img_resized / 255.0, 0, 1)
                        col1, col2 = st.columns(2)
                        col1.image(img_resized, caption="Extracted Frame")
                        col2.image(super_img, caption="Grad-CAM XAI Map")
                    else:
                        st.image(img_resized, caption="Extracted Frame (Grad-CAM unavailable)")
