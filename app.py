import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, Dense, Concatenate

st.set_page_config(page_title="Deepfake Detector XAI", layout="wide")

st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)

class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_forensic_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "MobileNetV2_best.keras")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, "models", "MobileNetV2_best.keras")

    try:
        raw = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
        
        # FIND BACKBONE
        backbone = None
        for layer in raw.layers:
            if 'mobilenet' in layer.name.lower():
                backbone = layer
                break
        if not backbone: backbone = raw.layers[0]

        # BUILD CLEAN GRAPH
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = backbone(inputs)

        # HANDLE THE "2 TENSORS" ERROR
        # If x is a list of two identical tensors, we merge them or take one
        if isinstance(x, list):
            if len(x) > 1:
                # Take the average or just the first to satisfy the Dense layer
                x = tf.keras.layers.Average()([x[0], x[1]])
            else:
                x = x[0]
        
        # Ensure spatial data is pooled to 1D
        if len(x.shape) > 2:
            x = GlobalAveragePooling2D()(x)
        
        outputs = Dense(1, activation='sigmoid')(x)
        fixed_model = tf.keras.Model(inputs, outputs)

        # COPY WEIGHTS FOR THE FINAL LAYER
        for layer in reversed(raw.layers):
            if 'dense' in layer.name.lower():
                fixed_model.layers[-1].set_weights(layer.get_weights())
                break
        
        return fixed_model
    except Exception as e:
        st.error(f"Critical Architecture Fix Failed: {e}")
        return None

model = load_forensic_model()

# --- Grad-CAM XAI Logic ---
def get_gradcam_heatmap(img_array, full_model):
    try:
        # Layer 1 is the backbone in our fixed model
        backbone = full_model.layers[1]
        last_conv = backbone.get_layer("out_relu")
        
        grad_model = tf.keras.Model(
            inputs=[backbone.input],
            outputs=[last_conv.output, backbone.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list): predictions = predictions[0]
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except:
        return None

# --- UI LOGIC ---
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(temp_path)
    
    if st.button("Generate Forensic Report", type="primary"):
        if model:
            with st.spinner('Analyzing...'):
                cap = cv2.VideoCapture(temp_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

                    preds = model.predict(img_array, verbose=0)
                    prediction = preds[0][0] if len(preds.shape) > 1 else preds[0]
                    
                    label = "AI (Deepfake)" if prediction < 0.5 else "Real Video"
                    conf = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

                    st.divider()
                    st.subheader(f"Verdict: {label} ({conf:.2f}%)")
                    
                    heatmap = get_gradcam_heatmap(img_array, model)
                    if heatmap is not None:
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_uint8 = np.uint8(255 * heatmap_resized)
                        color_map = cm.get_cmap("jet")
                        jet_heatmap = color_map(np.arange(256))[:, :3][heatmap_uint8]
                        super_img = np.clip(jet_heatmap * 0.4 + (img_resized / 255.0), 0, 1)
                        
                        c1, c2 = st.columns(2)
                        c1.image(img_resized, caption="Extracted Frame")
                        c2.image(super_img, caption="Grad-CAM XAI Map")
                else:
                    st.error("Frame extraction failed.")
