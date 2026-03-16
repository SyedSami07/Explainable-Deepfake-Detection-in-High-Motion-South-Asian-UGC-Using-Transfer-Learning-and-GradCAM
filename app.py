import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.cm as cm
import os

st.set_page_config(page_title="Deepfake Detector XAI", layout="wide")
st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)

# ── Patch InputLayer to accept legacy 'batch_shape' keyword ──────────────────
class LegacyInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]  # strip batch dim
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['input_shape'] = config.pop('batch_shape')[1:]
        return cls(**config)

@st.cache_resource
def load_forensic_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "MobileNetV2_best.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, "models", "MobileNetV2_best.keras")
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None

    try:
        raw = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={'InputLayer': LegacyInputLayer}
        )

        # Find MobileNetV2 backbone
        backbone = None
        for layer in raw.layers:
            if 'mobilenet' in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            st.error("MobileNetV2 backbone not found.")
            return None

        # Build clean functional model — handles dual-tensor backbone output
        inputs = tf.keras.Input(shape=(224, 224, 3))
        backbone_out = backbone(inputs, training=False)

        if isinstance(backbone_out, (list, tuple)):
            x = tf.keras.layers.Average()(list(backbone_out))
        else:
            x = backbone_out

        if len(x.shape) == 4:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)

        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Copy Dense weights from original model
        for layer in reversed(raw.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                try:
                    model.layers[-1].set_weights(layer.get_weights())
                    break
                except Exception:
                    pass

        return model

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


def get_gradcam_heatmap(img_array, model):
    try:
        backbone = None
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            return None

        last_conv = backbone.get_layer("out_relu")
        grad_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[last_conv.output, backbone.output]
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
                        jet_heatmap = cm.get_cmap("jet")(np.arange(256))[:, :3][heatmap_uint8]
                        super_img = np.clip(jet_heatmap * 0.4 + img_resized / 255.0, 0, 1)
                        col1, col2 = st.columns(2)
                        col1.image(img_resized, caption="Extracted Frame")
                        col2.image(super_img, caption="Grad-CAM XAI Map")
                    else:
                        st.image(img_resized, caption="Extracted Frame (Grad-CAM unavailable)")

