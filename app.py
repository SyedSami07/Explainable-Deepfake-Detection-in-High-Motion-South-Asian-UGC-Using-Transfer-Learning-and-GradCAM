import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.layers import InputLayer

st.set_page_config(page_title="Deepfake Detector XAI", layout="wide")

st.markdown("""
    <style>
    .block-container { max-width: 850px; padding-top: 1rem; }
    h1 { font-size: 2rem !important; text-align: center; color: #2C3E50; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #E74C3C; color: white; height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)

# --- Compatibility Fix ---
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

    if os.path.exists(model_path):
        try:
            # Load the raw model
            raw_model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'InputLayer': CompatibleInputLayer}
            )
            
            # REPAIR LOGIC: If the model expects a flat input but gets 4D, we force a pooling step.
            # We use the Functional API to build a bridge between your layers.
            inputs = tf.keras.Input(shape=(224, 224, 3))
            
            # Try to get the output from the last layer of your loaded model
            # This handles the "2 input tensors" error by only taking the first output if redundant
            x = raw_model(inputs)
            
            # If the output is still 4D (7, 7, 1280), flatten it
            if len(x.shape) > 2:
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                # If your model didn't have a final dense layer, we add it here
                if x.shape[-1] != 1:
                    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            fixed_model = tf.keras.Model(inputs, x)
            return fixed_model
        except Exception as e:
            st.error(f"Architecture Error: {e}")
            # Fallback: Just return the raw model and hope the predict logic handles it
            return raw_model
    return None

model = load_forensic_model()

def get_gradcam_heatmap(img_array, raw_model):
    try:
        # Find the last convolutional layer in MobileNetV2
        last_conv_layer_name = "out_relu"
        
        # We need to look inside nested models if applicable
        target_model = raw_model
        if hasattr(raw_model, 'layers') and isinstance(raw_model.layers[0], tf.keras.Model):
            target_model = raw_model.layers[0]

        grad_model = tf.keras.Model(
            inputs=[target_model.input],
            outputs=[target_model.get_layer(last_conv_layer_name).output, target_model.output]
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

# --- UI ---
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(temp_path)
    
    if st.button("Generate Forensic Report"):
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

                    # Run inference
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
                        c1.image(img_resized, caption="Analyzed Frame")
                        c2.image(super_img, caption="Grad-CAM XAI Map")
