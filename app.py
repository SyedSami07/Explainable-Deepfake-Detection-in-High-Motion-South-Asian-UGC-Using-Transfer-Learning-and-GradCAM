import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Deepfake Detector XAI", layout="wide")

st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)

@st.cache_resource
def load_forensic_model():
    # 1. Path Discovery
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(base_path, "models")
    
    # Try both common extensions
    target_file = "MobileNetV2_best.keras"
    model_path = os.path.join(model_folder, target_file)
    
    if not os.path.exists(model_path):
        target_file = "MobileNetV2_best.h5"
        model_path = os.path.join(model_folder, target_file)

    st.write(f"DEBUG: Looking for model at `{model_path}`") # Temporary visible debug

    if not os.path.exists(model_path):
        return None, None

    try:
        # 2. Keras 2/3 Compatibility Fix
        def fixed_input_layer(**kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            return tf.keras.layers.InputLayer(**kwargs)

        orig_model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={'InputLayer': fixed_input_layer}
        )
        
        # 3. Identify MobileNet Layer
        base_net = None
        for layer in orig_model.layers:
            if 'mobilenet' in layer.name.lower():
                base_net = layer
                break
        
        if base_net is None:
            base_net = orig_model.layers[0]
            
        # 4. Final Sequential Wrap
        new_model = tf.keras.Sequential([
            base_net,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return new_model, orig_model

    except Exception as e:
        st.error(f"Load Error: {str(e)}")
        return None, None

# Initialize Model
model, original_loaded_model = load_forensic_model()

# --- Helper for XAI ---
def get_gradcam_heatmap(img_array, raw_model):
    try:
        # Find the internal conv layer for MobileNetV2
        base_net = None
        for layer in raw_model.layers:
            if 'mobilenet' in layer.name.lower():
                base_net = layer
                break
        
        # Target the final ReLU in MobileNetV2 (out_relu)
        target_layer = base_net.get_layer("out_relu")
        
        grad_model = tf.keras.Model(
            inputs=[base_net.input],
            outputs=[target_layer.output, base_net.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
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
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(video_path)
    
    if st.button("Generate Forensic Report", type="primary"):
        if model:
            with st.spinner("Analyzing Video Frame..."):
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

                    # Prediction
                    prediction = model.predict(img_array, verbose=0)[0][0]
                    label = "AI (Deepfake)" if prediction < 0.5 else "Real Video"
                    conf = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

                    # XAI
                    heatmap = get_gradcam_heatmap(img_array, original_loaded_model)
                    
                    st.divider()
                    st.subheader(f"Verdict: {label} ({conf:.2f}%)")
                    
                    if heatmap is not None:
                        # Display Image + Heatmap
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_uint8 = np.uint8(255 * heatmap_resized)
                        color_map = cm.get_cmap("jet")
                        jet_heatmap = color_map(np.arange(256))[:, :3][heatmap_uint8]
                        super_img = np.clip(jet_heatmap * 0.4 + (img_resized / 255.0), 0, 1)
                        
                        c1, c2 = st.columns(2)
                        c1.image(img_resized, caption="Extracted Frame")
                        c2.image(super_img, caption="XAI Heatmap (Grad-CAM)")
        else:
            st.error("Model state is None. Please check if 'models/MobileNetV2_best.keras' exists in your GitHub repo.")
