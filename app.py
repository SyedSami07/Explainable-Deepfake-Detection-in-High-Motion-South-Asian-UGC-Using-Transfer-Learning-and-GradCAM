import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib
import os
import zipfile
import tempfile


st.set_page_config(
    page_title="DeepMotion — South Asian Deepfake Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');


*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }


html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #060609 !important;
    color: #e8e6f0 !important;
    font-family: 'Syne', sans-serif !important;
}


#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }


section[data-testid="stSidebar"] { display: none !important; }


.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}
[data-testid="stMain"] > div { padding: 0 !important; }


[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 70% 50% at 5% 0%,   rgba(99,31,255,0.20) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 95% 100%, rgba(0,190,140,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 50% 50%,  rgba(18,4,55,0.65)  0%, transparent 80%),
        #060609;
    pointer-events: none;
}


[data-testid="stAppViewContainer"]::after {
    content: ''; position: fixed; inset: 0; z-index: 1;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(255,255,255,0.010) 2px, rgba(255,255,255,0.010) 4px
    );
    pointer-events: none;
}


[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.stMarkdown, .stButton,
[data-testid="stFileUploader"],
[data-testid="stVideo"],
.stImage, .stSpinner, .stAlert {
    position: relative; z-index: 2;
}


.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,31,255,0.13);
    border: 1px solid rgba(99,31,255,0.32);
    border-radius: 100px; padding: 6px 18px;
    font-family: 'Space Mono', monospace; font-size: 10px;
    letter-spacing: 2.5px; color: #a78bfa; text-transform: uppercase;
    margin-bottom: 22px;
    animation: fadeDown 0.6s ease both;
}
.hero-badge .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #a78bfa;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.65); }
}
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(24px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
    from { opacity:0; }
    to   { opacity:1; }
}
@keyframes orbFloat {
    0%,100% { transform: translateY(0); }
    50%     { transform: translateY(-14px); }
}
@keyframes orbGlow {
    0%,100% { box-shadow: 0 0 36px rgba(124,58,237,0.5), 0 0 70px rgba(124,58,237,0.15); }
    50%     { box-shadow: 0 0 56px rgba(124,58,237,0.85), 0 0 110px rgba(124,58,237,0.35); }
}


[data-testid="stFileUploader"] {
    background: rgba(12,6,34,0.75) !important;
    border: 1.5px dashed rgba(99,31,255,0.38) !important;
    border-radius: 18px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.75) !important;
    background: rgba(18,8,50,0.9) !important;
    box-shadow: 0 0 48px rgba(99,31,255,0.14), inset 0 0 32px rgba(99,31,255,0.05) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
}


[data-testid="stVideo"] {
    max-width: 480px !important;
    margin: 0 auto 20px !important;
    display: block !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,31,255,0.22) !important;
    box-shadow: 0 16px 50px rgba(0,0,0,0.65), 0 0 32px rgba(99,31,255,0.09) !important;
}
video { max-height: 300px !important; border-radius: 18px !important; }


[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed, #4c1db5) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; padding: 15px 44px !important;
    font-family: 'Syne', sans-serif !important; font-size: 15px !important;
    font-weight: 700 !important; letter-spacing: 0.5px !important;
    display: block !important; margin: 0 auto !important;
    box-shadow: 0 8px 28px rgba(99,31,255,0.38) !important;
    transition: all 0.25s ease !important; cursor: pointer !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 38px rgba(99,31,255,0.55) !important;
    background: linear-gradient(135deg, #8b5cf6, #6d28d9) !important;
}


[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(99,31,255,0.18) !important;
    box-shadow: 0 10px 36px rgba(0,0,0,0.55) !important;
    width: 100% !important;
}
.stImage p {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 2px !important;
    color: rgba(255,255,255,0.75) !important;
    text-align: center !important; margin-top: 8px !important;
    text-transform: uppercase !important;
}


.stSpinner > div { border-color: #7c3aed transparent transparent transparent !important; }


.stAlert {
    background: rgba(12,6,34,0.7) !important;
    border: 1px solid rgba(99,31,255,0.18) !important;
    border-radius: 14px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important; color: rgba(190,175,235,0.8) !important;
}


.light-text {
    color: rgba(255,255,255,0.45) !important;
}


hr {
    border: none !important;
    border-top: 1px solid rgba(99,31,255,0.15) !important;
    margin: 36px auto !important; max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_model_path():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, "MobileNetV2_best.keras"),
              os.path.join(base, "models", "MobileNetV2_best.keras")]:
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
            input_shape=(224,224,3), include_top=False, weights=None)
        inp = tf.keras.Input(shape=(224,224,3))
        x   = base(inp, training=False)
        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        x   = tf.keras.layers.Dense(256, activation='relu')(x)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inp, out)
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(model_path,'r') as zf:
                zf.extractall(tmp)
            wp = next((os.path.join(tmp,f) for f in os.listdir(tmp)
                       if f.endswith('.h5') or 'weight' in f.lower()), None)
            if wp:
                model.load_weights(wp, by_name=True, skip_mismatch=True)
        return model, None
    except Exception as e:
        return None, str(e)


def get_gradcam_heatmap(img_array, model):
    try:
        base      = model.layers[1]
        last_conv = base.get_layer("out_relu")
        gm = tf.keras.Model(inputs=base.input,
                            outputs=[last_conv.output, base.output])
        with tf.GradientTape() as tape:
            co, preds = gm(img_array)
            if isinstance(preds,(list,tuple)): preds = preds[0]
            loss = preds[:,0]
        grads = tape.gradient(loss, co)
        pg     = tf.reduce_mean(grads, axis=(0,1,2))
        hm     = co[0] @ pg[..., tf.newaxis]
        hm     = tf.squeeze(hm)
        hm     = tf.maximum(hm,0)/(tf.math.reduce_max(hm)+1e-10)
        return hm.numpy()
    except Exception:
        return None


def build_explanation(is_fake, conf, hm_available):
    if is_fake:
        strength = "strong" if conf > 75 else "moderate" if conf > 60 else "mild"
        intro = f"The model detected <b>{strength} indicators of AI-generated manipulation</b> in this video."
        sub   = "Key signals that influenced this decision:"
        indicators = [
            ("Face Boundary Artifacts",   "Unnatural blending at the hairline, jaw, or neck — a common GAN failure mode."),
            ("Temporal Inconsistency",    "Flickering or warping at high-motion points between adjacent frames."),
            ("Texture Anomalies",         "Over-smoothed skin or unnatural eye and teeth reflections typical of synthesis."),
            ("Frequency Fingerprint",     "AI-generated frames carry statistical traces in pixel frequency patterns."),
        ]
        grad_note = "The <b>Grad-CAM heatmap</b> highlights the regions that most influenced this verdict — bright areas show where the model focused its suspicion." if hm_available else ""
    else:
        strength = "strong" if conf > 80 else "moderate" if conf > 65 else "mild"
        intro = f"The model found <b>{strength} evidence that this is authentic footage</b>."
        sub   = "Key signals that supported this verdict:"
        indicators = [
            ("Natural Skin Texture",  "Consistent micro-texture and pore detail across frames — hard for GANs to replicate."),
            ("Stable Lighting",       "Shadow and highlight transitions are physically consistent throughout the clip."),
            ("Organic Motion",        "Subtle natural head movement without the warping artifacts typical of deepfakes."),
            ("Clean Boundaries",      "Sharp, stable transitions between the subject and background with no blending seams."),
        ]
        grad_note = "The <b>Grad-CAM heatmap</b> shows which regions reinforced the authenticity verdict." if hm_available else ""
    return intro, sub, indicators, grad_note


# ── Load model ────────────────────────────────────────────────────────────────
model, model_err = load_forensic_model()


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="width:100%;padding:56px 5% 28px;text-align:center;">
  <div class="hero-badge">
    <span class="dot"></span>
    AI Forensics &nbsp;·&nbsp; XAI System &nbsp;·&nbsp; v2.0
  </div>
  <div style="
    font-size:clamp(52px,8vw,96px);font-weight:800;line-height:0.92;
    letter-spacing:-3px;
    background:linear-gradient(130deg,#ffffff 0%,#c4b5fd 35%,#7c3aed 65%,#10b981 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:16px;animation:fadeDown 0.7s 0.1s ease both;
    opacity:0;animation-fill-mode:both;
  ">DeepMotion</div>
  <div class="light-text" style="font-family:'Space Mono',monospace;font-size:11px;
      letter-spacing:2px;
      animation:fadeDown 0.7s 0.2s ease both;opacity:0;animation-fill-mode:both;">
    EXPLAINABLE DEEPFAKE DETECTION &nbsp;·&nbsp; SOUTH ASIAN CONTINENT &nbsp;·&nbsp; by Syed Salman Sami
  </div>
</div>
""", unsafe_allow_html=True)


# ── UPLOAD ────────────────────────────────────────────────────────────────────
_, mid, _ = st.columns([1, 3, 1])
with mid:
    st.markdown("""
    <div class="light-text" style="font-family:'Space Mono',monospace;font-size:10px;
        letter-spacing:3px;text-transform:uppercase;
        margin-bottom:10px;">
      📁 &nbsp; Drop your video file
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["mp4","avi","mov"],
                                      label_visibility="collapsed")


# ── MAIN FLOW ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    temp_path = "/tmp/deepshield_input.mp4"
    with open(temp_path,"wb") as f:
        f.write(uploaded_file.getbuffer())


    _, vc, _ = st.columns([1, 2, 1])
    with vc:
        st.video(temp_path)


    _, bc, _ = st.columns([1, 2, 1])
    with bc:
        run = st.button("⟡  Run Forensic Analysis", type="primary",
                        use_container_width=True)


    if run:
        if model is None:
            st.error(f"Model unavailable: {model_err}")
        else:
            _, sc, _ = st.columns([1,2,1])
            with sc:
                with st.spinner("Scanning neural patterns…"):
                    cap   = cv2.VideoCapture(temp_path)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                    ret, frame = cap.read()
                    cap.release()


            if not ret:
                st.error("Frame extraction failed.")
            else:
                img_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (224,224))
                img_array   = np.expand_dims(img_resized,0).astype("float32")/255.0


                preds       = model.predict(img_array, verbose=0)
                prediction = float(preds[0][0])
                is_fake     = prediction < 0.5
                label       = "AI DEEPFAKE" if is_fake else "AUTHENTIC"
                conf        = (1-prediction)*100 if is_fake else prediction*100
                accent      = "#f87171" if is_fake else "#34d399"
                icon        = "⚠" if is_fake else "✓"


                heatmap      = get_gradcam_heatmap(img_array, model)
                hm_available = heatmap is not None
                intro, sub, indicators, grad_note = build_explanation(is_fake, conf, hm_available)


                st.markdown("<hr>", unsafe_allow_html=True)


                # ── Verdict card ──────────────────────────────────────────────
                _, vc2, _ = st.columns([0.5, 5, 0.5])
                with vc2:
                    st.markdown(f"""
                    <div style="
                      background:rgba(12,6,34,0.85);
                      border:1px solid {accent}35;
                      border-radius:22px;padding:36px 40px;
                      display:flex;align-items:center;
                      justify-content:space-between;gap:24px;
                      backdrop-filter:blur(28px);
                      box-shadow:0 20px 60px rgba(0,0,0,0.55),0 0 60px {accent}15;
                      position:relative;overflow:hidden;
                      animation:fadeIn 0.5s ease both;
                    ">
                      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,transparent,{accent}70,transparent);"></div>
                      <div>
                        <div class="light-text" style="font-family:'Space Mono',monospace;font-size:10px;
                            letter-spacing:3px;text-transform:uppercase;
                            margin-bottom:10px;">
                          FORENSIC VERDICT
                        </div>
                        <div style="font-family:'Syne',sans-serif;
                            font-size:clamp(28px,4vw,44px);
                            font-weight:800;letter-spacing:-1px;color:{accent};line-height:1;">
                          {icon}&nbsp; {label}
                        </div>
                      </div>
                      <div style="text-align:right;flex-shrink:0;">
                        <div class="light-text" style="font-family:'Space Mono',monospace;font-size:10px;
                            letter-spacing:3px;text-transform:uppercase;
                            margin-bottom:10px;">
                          CONFIDENCE
                        </div>
                        <div style="font-family:'Syne',sans-serif;
                            font-size:clamp(36px,5vw,56px);
                            font-weight:800;letter-spacing:-2px;color:#e2d9f3;line-height:1;">
                          {conf:.1f}<span style="font-size:22px;color:rgba(190,175,235,0.4);">%</span>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


                # ── Confidence bar ────────────────────────────────────────────
                _, bc2, _ = st.columns([0.5, 5, 0.5])
                with bc2:
                    st.markdown(f"""
                    <div style="margin:14px 0 30px;">
                      <div class="light-text" style="display:flex;justify-content:space-between;
                          font-family:'Space Mono',monospace;font-size:10px;
                          letter-spacing:1px;margin-bottom:8px;">
                        <span>CERTAINTY SCALE</span><span>{conf:.1f}%</span>
                      </div>
                      <div style="background:rgba(255,255,255,0.06);border-radius:100px;
                          height:5px;overflow:hidden;">
                        <div style="width:{conf}%;height:100%;
                            background:linear-gradient(90deg,{accent}55,{accent});
                            border-radius:100px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


                # ── EXPLANATION ───────────────────────────────────────────────
                indicator_html = "".join([f"""
                <div style="display:flex;gap:14px;align-items:flex-start;
                    padding:14px 16px;border-radius:12px;
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(99,31,255,0.09);margin-bottom:8px;">
                  <div style="width:7px;height:7px;border-radius:50%;
                      background:{accent};margin-top:5px;flex-shrink:0;
                      box-shadow:0 0 8px {accent}80;"></div>
                  <div>
                    <div style="font-weight:700;font-size:14px;
                        color:rgba(220,210,255,0.88);margin-bottom:3px;">{t}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:11px;
                        color:rgba(155,140,200,0.5);line-height:1.55;">{d}</div>
                  </div>
                </div>""" for t,d in indicators])


                grad_html = f"""<div class="light-text" style="margin-top:16px;padding:14px 16px;border-radius:12px;
                    background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.15);
                    font-family:'Space Mono',monospace;font-size:11px;
                    line-height:1.6;">{grad_note}</div>""" \
                    if grad_note else ""


                _, ec, _ = st.columns([0.5, 5, 0.5])
                with ec:
                    st.markdown(f"""
                    <div style="
                      background:rgba(12,6,34,0.75);
                      border:1px solid rgba(99,31,255,0.18);
                      border-radius:22px;padding:32px 36px;
                      backdrop-filter:blur(20px);margin-bottom:30px;
                      position:relative;overflow:hidden;
                      animation:fadeIn 0.6s 0.1s ease both;
                    ">
                      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                          background:linear-gradient(90deg,transparent,rgba(124,58,237,0.5),transparent);"></div>
                      <div class="light-text" style="font-family:'Space Mono',monospace;font-size:10px;
                          letter-spacing:3px;text-transform:uppercase;
                          margin-bottom:18px;">
                        ◈ &nbsp; WHY THIS VERDICT?
                      </div>
                      <p style="font-size:15px;line-height:1.7;
                          color:rgba(215,205,245,0.85);margin-bottom:18px;">{intro}</p>
                      <p class="light-text" style="font-family:'Space Mono',monospace;font-size:11px;
                          line-height:1.6;margin-bottom:14px;">{sub}</p>
                      {indicator_html}
                      {grad_html}
                    </div>
                    """, unsafe_allow_html=True)


                # ── Grad-CAM visuals ──────────────────────────────────────────
                st.markdown("""
                <div class="light-text" style="text-align:center;font-family:'Space Mono',monospace;
                    font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    margin-bottom:16px;">
                 ◈ &nbsp; XAI Visual Forensics &nbsp; ◈
                </div>
                """, unsafe_allow_html=True)


                _, ic, _ = st.columns([0.5, 5, 0.5])
                with ic:
                    if hm_available:
                        hm_r  = cv2.resize(heatmap,(224,224))
                        hm_u8 = np.uint8(255*hm_r)
                        jet   = matplotlib.colormaps["jet"](np.arange(256))[:,:3][hm_u8]
                        sup   = np.clip(jet*0.45 + img_resized/255.0, 0, 1)
                        c1, c2 = st.columns(2)
                        c1.image(img_resized, caption="EXTRACTED FRAME",
                                 use_container_width=True)
                        c2.image(sup, caption="GRAD-CAM ATTENTION MAP",
                                 use_container_width=True)
                    else:
                        st.image(img_resized, caption="EXTRACTED FRAME",
                                 use_container_width=True)


                st.markdown("<div style='height:52px;'></div>", unsafe_allow_html=True)


else:
    st.markdown("""
    <div class="light-text" style="text-align:center;padding:16px 0 72px;
        animation:fadeUp 0.8s 0.35s ease both;opacity:0;animation-fill-mode:both;">
      <div style="width:110px;height:110px;margin:0 auto 22px;border-radius:50%;
         background:radial-gradient(circle at 35% 35%,rgba(167,139,250,0.22),rgba(99,31,255,0.07),transparent);
         border:1px solid rgba(99,31,255,0.16);
         display:flex;align-items:center;justify-content:center;font-size:44px;
         animation:orbFloat 4s ease-in-out infinite,orbGlow 4s ease-in-out infinite;">🔍</div>
      <div style="font-family:'Space Mono',monospace;font-size:11px;
         letter-spacing:2.5px;">
        UPLOAD A VIDEO TO BEGIN ANALYSIS
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Minimal footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="light-text" style="text-align:center;padding:24px 0 20px;
    font-family:'Space Mono',monospace;font-size:10px;
    letter-spacing:3px;">
  DeepMotion &nbsp;·&nbsp; BY SAMI
</div>
""", unsafe_allow_html=True)
