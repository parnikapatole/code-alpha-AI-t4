import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import av
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionTrac Ai | Helly", layout="wide")

# --- CUSTOM UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Bree+Serif&display=swap');
    .stApp { background-color: #000000; color: white; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    .main-title {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 28px !important; 
        color: #00d2ff !important;
        margin-left: 50px; margin-top: 40px;
    }
    .bottom-container { margin-left: 50px; margin-top: 30px; }
    div.stButton > button:first-child {
        background-color: #00d2ff; color: #000; font-weight: bold;
        padding: 15px 30px; border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- LANDING PAGE ---
if not st.session_state.entered:
    st.markdown('<h1 class="main-title">VisionTrac AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 20px; color: #888; margin-bottom: 30px;">Engineered by Helly</div>', unsafe_allow_html=True)
    if st.button("Launch Engine"):
        st.session_state.entered = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- DETECTION PAGE ---
else:
    with st.sidebar:
        st.title("⚙️ Settings")
        app_mode = st.radio("Device Mode", ["PC Mode", "Android/Mobile"])
        camera_choice = st.selectbox("Select Camera", ["Back Camera (Environment)", "Front Camera (User)"])
        facing_mode = "environment" if "Back" in camera_choice else "user"
        st.divider()
        if st.button("← Exit Engine"):
            st.session_state.entered = False
            st.rerun()

    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt') # Ensure the .pt extension is included

    model = load_model()

    st.markdown('<p style="font-family: \'Bree Serif\', serif; font-size: 42px; color: #00d2ff;">VisionTrac Ai: Real-Time</p>', unsafe_allow_html=True)

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Run inference without 'stream=True' for single-frame callback stability
        results = model.predict(img, conf=0.4, imgsz=320, verbose=False)
        
        # Plot results onto the frame if detected
        annotated_frame = results[0].plot() if results else img
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Use Google's public STUN server for connection establishment
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key=f"vision-engine-{facing_mode}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": facing_mode,
                "width": {"ideal": 1280 if app_mode == "PC Mode" else 640},
                "height": {"ideal": 720 if app_mode == "PC Mode" else 480},
            },
            "audio": False
        },
        async_processing=True,
    )
