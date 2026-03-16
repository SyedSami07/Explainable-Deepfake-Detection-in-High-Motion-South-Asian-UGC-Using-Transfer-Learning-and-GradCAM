import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
import os
import zipfile
import tempfile

st.set_page_config(
    page_title="DeepShield — Deepfake Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject premium dark UI ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #050508 !important;
    color: #e8e6f0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* ── Main container ── */
[data-testid="stMain"] > div { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { display: none !important; }

/* ── Background mesh gradient ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(99,31,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(0,200,150,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(20,5,60,0.6) 0%, transparent 80%),
        #050508;
    pointer-events: none;
    z-index: 0;
}

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.012) 2px,
        rgba(255,255,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 1;
}

/* ── Everything above overlays ── */
[data-testid="stVerticalBlock"], .stMarkdown, .stButton,
[data-testid="stFileUploader"], .stVideo, .stImage,
[data-testid="metric-container"], .stSpinner, .stAlert,
[data-testid="stHorizontalBlock"] {
    position: relative;
    z-index: 2;
}

/* ── Hero Header ── */
.hero-wrap {
    width: 100%;
    padding: 60px 5% 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,31,255,0.15);
    border: 1px solid rgba(99,31,255,0.35);
    border-radius: 100px;
    padding: 6px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    color: #a78bfa;
    text-transform: uppercase;
    margin-bottom: 24px;
    animation: fadeSlideDown 0.6s ease both;
}

.hero-badge::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #a78bfa;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

.hero-title {
    font-size: clamp(42px, 7vw, 88px);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -3px;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #7c3aed 70%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 18px;
    animation: fadeSlideDown 0.7s 0.1s ease both;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: rgba(200,190,230,0.55);
    letter-spacing: 1px;
    animation: fadeSlideDown 0.7s 0.2s ease both;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Upload Zone ── */
.upload-zone-wrap {
    width: 100%;
    max-width: 760px;
    margin: 0 auto 48px;
    padding: 0 20px;
    animation: fadeUp 0.7s 0.3s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.upload-label {
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(167,139,250,0.7);
    margin-bottom: 12px;
    display: block;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 8, 40, 0.7) !important;
    border: 1.5px dashed rgba(99,31,255,0.4) !important;
    border-radius: 20px !important;
    padding: 32px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.8) !important;
    background: rgba(20, 10, 55, 0.85) !important;
    box-shadow: 0 0 40px rgba(99,31,255,0.15), inset 0 0 40px rgba(99,31,255,0.05) !important;
}

[data-testid="stFileUploader"] label {
    color: rgba(200,185,250,0.7) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* Upload icon area */
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(167,139,250,0.6) !important;
}

/* ── Video Player ── */
.video-panel {
    max-width: 560px;
    margin: 0 auto 32px;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(99,31,255,0.25);
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(99,31,255,0.1);
    animation: fadeUp 0.5s ease both;
}

[data-testid="stVideo"] {
    max-width: 560px !important;
    margin: 0 auto 32px !important;
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,31,255,0.25) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(99,31,255,0.1) !important;
    display: block !important;
}

video {
    max-height: 320px !important;
    border-radius: 20px !important;
}

/* ── Analyze Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4f1db5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 48px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    display: block !important;
    margin: 0 auto !important;
    box-shadow: 0 8px 32px rgba(99,31,255,0.4) !important;
    position: relative !important;
    overflow: hidden !important;
    width: auto !important;
}

[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.5s ease;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 40px rgba(99,31,255,0.6) !important;
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
}

[data-testid="stButton"] > button:hover::before { left: 100%; }

/* ── Results Grid ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    max-width: 760px;
    margin: 0 auto 32px;
    padding: 0 20px;
    animation: fadeUp 0.6s ease both;
}

.result-card {
    background: rgba(15,8,40,0.7);
    border: 1px solid rgba(99,31,255,0.2);
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.6), transparent);
}

.result-card .rc-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(167,139,250,0.6);
    margin-bottom: 10px;
}

.result-card .rc-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
}

.rc-deepfake { color: #f87171; }
.rc-real     { color: #34d399; }
.rc-conf     { color: #a78bfa; }

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: rgba(15,8,40,0.7) !important;
    border: 1px solid rgba(99,31,255,0.2) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(167,139,250,0.6) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #e2d9f3 !important;
}

/* ── Image panels ── */
[data-testid="stImage"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,31,255,0.2) !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
}

[data-testid="stImage"] img {
    border-radius: 16px !important;
}

.stImage > div > p {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: rgba(167,139,250,0.5) !important;
    letter-spacing: 1px !important;
    text-align: center !important;
    margin-top: 8px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(99,31,255,0.2) !important;
    margin: 32px auto !important;
    max-width: 760px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #7c3aed transparent transparent transparent !important;
}

/* ── Alerts ── */
.stAlert {
    background: rgba(15,8,40,0.7) !important;
    border: 1px solid rgba(99,31,255,0.2) !important;
    border-radius: 14px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    color: rgba(200,185,250,0.8) !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ── Warning override ── */
.stAlert[data-baseweb="notification"] {
    background: rgba(15,8,40,0.7) !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(167,139,250,0.5);
    text-align: center;
    margin: 24px 0 16px;
}

/* ── Columns spacing ── */
[data-testid="stHorizontalBlock"] {
    max-width: 760px;
    margin: 0 auto;
    gap: 20px;
}

/* ── Loading animation overlay ── */
.loading-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    padding: 40px;
    max-width: 400px;
    margin: 0 auto;
}

.loading-orb {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #a78bfa, #7c3aed, #1e0060);
    box-shadow: 0 0 40px rgba(124,58,237,0.6), 0 0 80px rgba(124,58,237,0.2);
    animation: orbFloat 3s ease-in-out infinite, orbGlow 3s ease-in-out infinite;
    position: relative;
}

.loading-orb::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: rgba(167,139,250,0.6);
    animation: spin 1.5s linear infinite;
}

@keyframes orbFloat {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-12px); }
}

@keyframes orbGlow {
    0%, 100% { box-shadow: 0 0 40px rgba(124,58,237,0.6), 0 0 80px rgba(124,58,237,0.2); }
    50%       { box-shadow: 0 0 60px rgba(124,58,237,0.9), 0 0 120px rgba(124,58,237,0.4); }
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ── Bottom credit ── */
.credit {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: rgba(120,100,180,0.35);
    letter-spacing: 2px;
    padding: 40px 0 30px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────────
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
        return None, "Model file not found."
    try:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None)
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)

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
        return model, None
    except Exception as e:
        return None, str(e)

def get_gradcam_heatmap(img_array, model):
    try:
        base = model.layers[1]
        last_conv = base.get_layer("out_relu")
        grad_model = tf.keras.Model(
            inputs=base.input, outputs=[last_conv.output, base.output])
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
    except Exception:
        return None

# ── Load model silently ───────────────────────────────────────────────────────
model, model_err = load_forensic_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">⬡ AI Forensics · XAI System · v2.0</div>
  <div class="hero-title">DeepShield</div>
  <div class="hero-sub">EXPLAINABLE DEEPFAKE DETECTION &nbsp;·&nbsp; GRAD-CAM VISUALIZATION &nbsp;·&nbsp; by Sami</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
_, col_upload, _ = st.columns([1, 2, 1])
with col_upload:
    st.markdown('<span class="upload-label">📁 &nbsp; Drop your video file</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "", type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

# ── Video + Analysis ──────────────────────────────────────────────────────────
if uploaded_file is not None:
    temp_path = "/tmp/deepshield_input.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Video — constrained
    _, vcol, _ = st.columns([1, 2, 1])
    with vcol:
        st.video(temp_path)

    # Button
    _, bcol, _ = st.columns([1, 2, 1])
    with bcol:
        run = st.button("⟡  Run Forensic Analysis", type="primary", use_container_width=True)

    if run:
        if model is None:
            st.error(f"Model unavailable: {model_err}")
        else:
            # Loading animation
            _, lcol, _ = st.columns([1, 2, 1])
            with lcol:
                with st.spinner("Scanning neural patterns…"):
                    cap = cv2.VideoCapture(temp_path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                    ret, frame = cap.read()
                    cap.release()

            if not ret:
                st.error("Frame extraction failed.")
            else:
                img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array   = np.expand_dims(img_resized, axis=0).astype("float32") / 255.0

                preds      = model.predict(img_array, verbose=0)
                prediction = float(preds[0][0])
                is_fake    = prediction < 0.5
                label      = "AI DEEPFAKE" if is_fake else "AUTHENTIC"
                conf       = (1 - prediction) * 100 if is_fake else prediction * 100
                accent     = "#f87171" if is_fake else "#34d399"
                icon       = "⚠" if is_fake else "✓"

                # ── Verdict card ─────────────────────────────────────────────
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="max-width:760px;margin:0 auto 28px;padding:0 20px;">
                  <div style="
                    background: rgba(15,8,40,0.85);
                    border: 1px solid {accent}40;
                    border-radius: 24px;
                    padding: 36px 40px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 24px;
                    backdrop-filter: blur(30px);
                    box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 60px {accent}18;
                    position: relative; overflow: hidden;
                  ">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,transparent,{accent}80,transparent);"></div>
                    <div>
                      <div style="font-family:'Space Mono',monospace;font-size:10px;
                          letter-spacing:3px;text-transform:uppercase;
                          color:rgba(200,185,250,0.5);margin-bottom:10px;">
                        FORENSIC VERDICT
                      </div>
                      <div style="font-family:'Syne',sans-serif;font-size:38px;
                          font-weight:800;letter-spacing:-1px;color:{accent};">
                        {icon}&nbsp; {label}
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-family:'Space Mono',monospace;font-size:10px;
                          letter-spacing:3px;text-transform:uppercase;
                          color:rgba(200,185,250,0.5);margin-bottom:10px;">
                        CONFIDENCE
                      </div>
                      <div style="font-family:'Syne',sans-serif;font-size:48px;
                          font-weight:800;letter-spacing:-2px;color:#e2d9f3;">
                        {conf:.1f}<span style="font-size:24px;color:rgba(200,185,250,0.5)">%</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence bar ───────────────────────────────────────────
                bar_pct = conf
                bar_col = accent
                st.markdown(f"""
                <div style="max-width:760px;margin:0 auto 36px;padding:0 20px;">
                  <div style="background:rgba(255,255,255,0.06);border-radius:100px;
                      height:6px;overflow:hidden;">
                    <div style="width:{bar_pct}%;height:100%;
                        background:linear-gradient(90deg,{bar_col}80,{bar_col});
                        border-radius:100px;
                        transition:width 1s cubic-bezier(.4,0,.2,1);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Grad-CAM ─────────────────────────────────────────────────
                heatmap = get_gradcam_heatmap(img_array, model)

                st.markdown('<div class="section-label">◈ &nbsp; XAI Visual Forensics &nbsp; ◈</div>',
                            unsafe_allow_html=True)

                _, imgcol, _ = st.columns([0.15, 2, 0.15])
                with imgcol:
                    if heatmap is not None:
                        hm_resized  = cv2.resize(heatmap, (224, 224))
                        hm_uint8    = np.uint8(255 * hm_resized)
                        jet_heatmap = matplotlib.colormaps["jet"](np.arange(256))[:, :3][hm_uint8]
                        super_img   = np.clip(jet_heatmap * 0.45 + img_resized / 255.0, 0, 1)

                        c1, c2 = st.columns(2)
                        c1.image(img_resized,
                                 caption="EXTRACTED FRAME",
                                 use_container_width=True)
                        c2.image(super_img,
                                 caption="GRAD-CAM ATTENTION MAP",
                                 use_container_width=True)
                    else:
                        st.image(img_resized,
                                 caption="EXTRACTED FRAME",
                                 use_container_width=True)

                # ── Footer note ───────────────────────────────────────────────
                st.markdown(f"""
                <div style="max-width:760px;margin:28px auto 0;padding:0 20px;">
                  <div style="
                    background: rgba(15,8,40,0.5);
                    border: 1px solid rgba(99,31,255,0.15);
                    border-radius: 16px;
                    padding: 18px 24px;
                    display: flex;
                    align-items: center;
                    gap: 14px;
                    font-family: 'Space Mono', monospace;
                    font-size: 11px;
                    color: rgba(167,139,250,0.5);
                    letter-spacing: 0.5px;
                  ">
                    <span style="font-size:18px;">⬡</span>
                    MobileNetV2 transfer learning · Grad-CAM explainability ·
                    Frame sampled at 50% video position · Confidence threshold: 50%
                  </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:20px 0 60px;animation:fadeUp 0.8s 0.4s ease both;opacity:0;
        animation-fill-mode:both;">
      <div style="
        width: 120px; height: 120px;
        margin: 0 auto 24px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, rgba(167,139,250,0.3), rgba(99,31,255,0.1), transparent);
        border: 1px solid rgba(99,31,255,0.2);
        display: flex; align-items: center; justify-content: center;
        font-size: 48px;
        box-shadow: 0 0 60px rgba(99,31,255,0.1);
        animation: orbFloat 4s ease-in-out infinite;
      ">🔍</div>
      <div style="font-family:'Space Mono',monospace;font-size:12px;
          color:rgba(167,139,250,0.4);letter-spacing:2px;">
        UPLOAD A VIDEO TO BEGIN ANALYSIS
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Credit ────────────────────────────────────────────────────────────────────
st.markdown('<div class="credit">DEEPSHIELD &nbsp;·&nbsp; FORENSIC AI TOOL &nbsp;·&nbsp; BY SAMI</div>',
            unsafe_allow_html=True)
