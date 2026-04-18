import streamlit as st
import pandas as pd
import base64, os
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# ===============================
# 🎞️ Background Video Function
# ===============================
def add_bg_video(video_file):
    if os.path.exists(video_file):
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        video_html = f"""
        <style>
        .stApp {{
            background: none;
        }}
        #background-video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
            opacity: 0.75;
        }}
        .glass-box {{
            background: rgba(255, 255, 255, 0.55);
            border-radius: 15px;
            padding: 15px 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #000;
            display: inline-block;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .dark-glass {{
            background: rgba(0, 0, 0, 0.55);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #fff;
            text-align: center;
            margin-top: 25px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
            display: inline-block;
        }}
        .center-box {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}
        .dark-caption {{
            background: rgba(0, 0, 0, 0.6);
            color: #fff;
            padding: 6px 14px;
            border-radius: 12px;
            display: inline-block;
            margin-top: 8px;
            font-size: 15px;
            font-weight: 500;
        }}
        </style>
        <video id="background-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Background video '{video_file}' not found!")

# ===============================
# ⚙️ Page Setup
# ===============================
st.set_page_config(page_title="Ocean Plastic Detection", layout="wide")

# Background Video
add_bg_video("sea_vid.mp4")

# ===============================
# 🌊 Title + Logo
# ===============================
with open("logo.jpg", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 10px;
    ">
        <img src="data:image/jpg;base64,{logo_base64}" 
             width="60" 
             style="border-radius: 50%; box-shadow: 0 0 10px rgba(255,255,255,0.4);">
        <h1 style="margin: 0; font-size: 42px; color: white;">VarunX Plastic Detection</h1>
    </div>
""", unsafe_allow_html=True)

# ===============================
# 📋 Description
# ===============================
st.markdown("""
<div class='center-box'>
    <div class='glass-box'>
        <p><b>Upload an image to detect plastic using our trained YOLO model.</b></p>
        <p><b>The app will show detection results and average detection accuracy (%)</b></p>
    </div>
</div>
""", unsafe_allow_html=True)

# ===============================
# 🧠 Load YOLO Model
# ===============================
MODEL_PATH = r"D:\\codes\\python\\Varun_x\\adit.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at: {MODEL_PATH}")
    st.stop()
else:
    model = YOLO(MODEL_PATH)

# ===============================
# 📊 Initialize Clean DataFrame
# ===============================
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["date", "type", "image_data"])

# ===============================
# 📸 Image Upload
# ===============================
uploaded_image = st.file_uploader(" ", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.markdown("<div class='center-box' style='margin-top:20px;'>", unsafe_allow_html=True)
    st.image(uploaded_image, width=400)
    st.markdown("<div class='dark-caption'>Uploaded Image</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    with st.spinner("🔍 Detecting plastics..."):
        results = model.predict(source=temp_path, save=True, conf=0.4)
        detected_image_path = os.path.join(results[0].save_dir, os.path.basename(results[0].path))

    st.markdown("<div class='center-box'><div class='glass-box'>✅ Detection complete!</div></div>",
                unsafe_allow_html=True)

    st.markdown("<div class='center-box' style='margin-top:30px;'>", unsafe_allow_html=True)
    st.image(str(detected_image_path), width=500)
    st.markdown("<div class='dark-caption'>Detected Plastics</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    confidences = [float(box.conf) for box in results[0].boxes]
    accuracy_percent = sum(confidences) / len(confidences) * 100 if confidences else None

    boxes = results[0].boxes
    detected_objects = [f"{results[0].names[int(box.cls)]} ({float(box.conf):.2f})" for box in boxes]
    total_detected = len(detected_objects)

    if accuracy_percent is not None:
        st.markdown(f"""
        <div class='center-box'>
            <div class='dark-glass'>
                <h3>Detected Plastics Summary</h3>
                <p>📈 Average Detection Accuracy: <b>{accuracy_percent:.2f}%</b></p>
                <p>🧩 Total Objects Detected: <b>{total_detected}</b></p>
                <p><b>Detected objects:</b> {', '.join(detected_objects)}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='center-box'>
            <div class='dark-glass'>⚠️ No plastics detected in the uploaded image.</div>
        </div>
        """, unsafe_allow_html=True)

    with open(detected_image_path, "rb") as img_f:
        b64_img = base64.b64encode(img_f.read()).decode("utf-8")
    img_url = f"data:image/jpeg;base64,{b64_img}"

    new_event = {
        "date": datetime.today().date().isoformat(),
        "type": detected_objects[0] if detected_objects else "Unknown",
        "image_data": img_url
    }

    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_event])], ignore_index=True)

# ===============================
# 📋 Event Log
# ===============================
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>🗂️ Recent Detection Events</h3>", unsafe_allow_html=True)

if not st.session_state.data.empty:
    st.dataframe(st.session_state.data.drop(columns=["image_data"]))
else:
    st.markdown("""
    <div class='center-box'>
        <div class='glass-box'><b>No detections yet.</b></div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# 💾 Centered Download Button
# ===============================
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 0.35, 1])
with col2:
    st.download_button(
        label="💾 Download Detection Data (CSV)",
        data=st.session_state.data.to_csv(index=False),
        file_name="detection_data.csv",
        mime="text/csv"
    )

st.markdown(
    "<p style='text-align:center;'>Built with <b>YOLOv12 + Streamlit</b> | Ocean Plastic Detection Hackathon</p>",
    unsafe_allow_html=True
)
