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

st.markdown("""
    <style>
    .block-container { max-width: 850px; padding-top: 1rem; }
    h1 { font-size: 2rem !important; text-align: center; color: #2C3E50; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #E74C3C; color: white; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Deepfake Detection with XAI")
st.write("<p style='text-align: center;'>Forensic Tool - by Sami</p>", unsafe_allow_html=True)


@st.cache_resource
def load_and_fix_model():
    # Get the absolute path to the directory where app.py lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Join it with the models folder and filename
    model_path = os.path.join(current_dir, "models", "MobileNetV2_best.h5")
    
    if os.path.exists(model_path):
        try:
            orig_model = tf.keras.models.load_model(model_path, compile=False)
            # ... rest of your code ...
            return new_model, orig_model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    else:
        # This will help you see exactly where the app is looking
        st.error(f"File not found. Searched at: {model_path}")
        return None, None

model, original_loaded_model = load_and_fix_model()


def get_gradcam_heatmap(img_array, full_model, raw_model, last_conv_layer_name="out_relu"):
    try:
        
        base_net = None
        for layer in raw_model.layers:
            if 'mobilenet' in layer.name.lower():
                base_net = layer
                break
        
        if not base_net:
            base_net = raw_model.layers[0]

       
        inner_grad_model = tf.keras.Model(
            inputs=[base_net.input],
            outputs=[base_net.get_layer(last_conv_layer_name).output, base_net.output]
        )

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = inner_grad_model(img_tensor)
           
            loss = predictions[:, 0]

      
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
     
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
       
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"XAI Detail: {e}")
        return None


def create_pdf(label, confidence, video_name, orig_path, heat_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 15, txt="Deepfake Forensic Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"File: {video_name}", ln=True)
    pdf.cell(0, 10, txt=f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, txt=f"Final Verdict: {label} ({confidence:.2f}%)", ln=True)
    
  
    pdf.ln(10)
    pdf.image(orig_path, x=15, y=70, w=85)
    pdf.image(heat_path, x=110, y=70, w=85)
    
    
    pdf.set_y(150)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Note: Heatmaps indicate areas of high interest for the AI model. "
                             "Red zones suggest spatial artifacts often found in synthetic manipulations.")
    
    return pdf.output(dest='S').encode('latin-1')


uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col_l, col_m, col_r = st.columns([1, 1, 1]) 
    with col_m:
        st.write("### Preview")
        st.video("temp_video.mp4")
        analyze_btn = st.button("Generate Forensic Report")
    
    if analyze_btn:
        if model:
           
            _ = model(np.zeros((1, 224, 224, 3)))
            
            with st.spinner('Analyzing high-motion artifacts...'):
                cap = cv2.VideoCapture("temp_video.mp4")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (224, 224))
                    img_array = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

                 
                    prediction = model.predict(img_array, verbose=0)[0][0]
                    
                    label = "AI (Deepfake)" if prediction < 0.5 else "Real Video"
                    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100

                  
                    heatmap = get_gradcam_heatmap(img_array, model, original_loaded_model)

                    if heatmap is not None:
                        heatmap_resized = cv2.resize(heatmap, (224, 224))
                        heatmap_uint8 = np.uint8(255 * heatmap_resized)
                        
                       
                        color_map = cm.get_cmap("jet")
                        jet_heatmap = color_map(np.arange(256))[:, :3][heatmap_uint8]
                        
                       
                        super_img = np.clip(jet_heatmap * 0.4 + (img_resized / 255.0), 0, 1)

                     
                        plt.imsave("orig.png", img_resized)
                        plt.imsave("heat.png", super_img)

                        st.divider()
                        c1, c2 = st.columns(2)
                        with c1: st.image(img_resized, caption="Extracted Frame")
                        with c2: st.image(super_img, caption="Grad-CAM XAI Heatmap")
                        
                        st.subheader(f"Verdict: {label} ({confidence:.2f}%)")
                        
                        pdf_data = create_pdf(label, confidence, uploaded_file.name, "orig.png", "heat.png")
                        st.download_button("📥 Download Report", pdf_data, "Forensic_Report.pdf", "application/pdf")
                    else:
                        st.warning("Heatmap failed. Displaying Verdict only.")
                        st.subheader(f"Verdict: {label} ({confidence:.2f}%)")
