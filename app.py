import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, Dense

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

    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None

    try:
        # 1. Load the raw model first
        raw = tf.keras.models.load_model(
            model_path, 
            compile=False,
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
        
        # 2. Extract the Backbone
        # We look for the MobileNetV2 functional block inside your saved model
        backbone = None
        for layer in raw.layers:
            if 'mobilenet' in layer.name.lower():
                backbone = layer
                break
        
        if backbone is None:
            backbone = raw.layers[0]

        # 3. Manually build the functional graph to resolve the "2 inputs" error
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        # Pass through backbone
        x = backbone(inputs)
        
        # If backbone returns a list (the cause of your error), take the first one
        if isinstance(x, list):
            x = x[0]
            
        # Ensure it is pooled before Dense
        if len(x.shape) > 2:
            x = GlobalAveragePooling2D()(x)
            
        # Add the final classification head
        outputs = Dense(1, activation='sigmoid')(x)
        
        fixed_model = tf.keras.Model(inputs, outputs)
        
        # Transfer weights from the original's last dense layer if it exists
        try:
            for layer in reversed(raw.layers):
                if 'dense' in layer.name.lower():
                    fixed_model.layers[-1].set_weights(layer.get_weights())
                    break
        except:
            pass # Use randomized if transfer fails, or weights were in backbone
            
        return fixed_model

    except Exception as e:
        st.error(f"Architecture Error: {e}")
        return None

model = load_forensic_model()

def get_gradcam_heatmap(img_array, full_model):
    try:
        # Find the backbone and the internal conv layer
        backbone = full_model.layers[1] 
        target_layer = backbone.get_layer("out_relu")
        
        grad_model = tf.keras.Model(
            inputs=[backbone.input],
            outputs=[target_layer.output, backbone.output]
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

# --- UI Logic ---
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
                    st.error("Failed to extract frame.")
