import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import av
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="VisionTrac Ai", layout="wide")

# --- CUSTOM UI & BACKGROUND ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Bree+Serif&display=swap');

    /* Pure Black Background */
    .stApp {
        background-color: #000000;
        color: white;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333;
    }

    .main-title {
        font-family: 'Press Start 2P', cursive !important;
        font-size: 28px !important; 
        color: #00d2ff !important;
        text-shadow: 2px 2px 5px rgba(0, 210, 255, 0.3);
        text-align: left !important;
        margin-left: 50px;
        margin-top: 40px;
        display: block;
    }

    .bottom-container {
        text-align: left !important;
        margin-left: 50px;
        margin-top: 30px;
        width: 100%;
    }
    
    div.stButton > button:first-child {
        background-color: #00d2ff;
        color: #000;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 8px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'entered' not in st.session_state:
    st.session_state.entered = False

# --- LANDING PAGE ---
if not st.session_state.entered:
    st.markdown('<h1 class="main-title">VisionTrac AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 20px; color: #888; margin-bottom: 30px;">Deep Learning Vision Engine</div>', unsafe_allow_html=True)
    if st.button("Launch Engine"):
        st.session_state.entered = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- DETECTION PAGE ---
else:
    # --- SETTINGS SIDEBAR ---
    with st.sidebar:
        st.title("⚙️ Settings")
        app_mode = st.radio("Device Mode", ["PC Mode", "Android/Mobile"])
        
        # Camera selection logic
        camera_choice = st.selectbox("Select Camera", ["Back Camera (Environment)", "Front Camera (User)"])
        facing_mode = "environment" if "Back" in camera_choice else "user"
        
        st.divider()
        if st.button("← Exit Engine"):
            st.session_state.entered = False
            st.rerun()

    # Load Model
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n')

    model = load_model()

    st.markdown('<p style="font-family: \'Bree Serif\', serif; font-size: 42px; color: #00d2ff;">VisionTrac Ai: Real-Time</p>', unsafe_allow_html=True)
    st.info(f"Currently optimized for: {app_mode}")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.4, imgsz=320, stream=True)
        annotated_frame = img
        for r in results:
            annotated_frame = r.plot()
            break 
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Key includes facing_mode so it resets the component when you switch cameras
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