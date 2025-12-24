import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Constants & Setup ---
MODEL_PATH = 'pose_landmarker.task'

# Landmark indices for MediaPipe Pose
# We will focus on the main skeleton for verified visualization
CONNECTIONS = [
    (11, 12), # Shoulders
    (11, 13), (13, 15), # Left Arm
    (12, 14), (14, 16), # Right Arm
    (11, 23), (12, 24), # Torso
    (23, 24), # Hips
    (23, 25), (25, 27), # Left Leg
    (24, 26), (26, 28)  # Right Leg
]

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c).
    b is the vertex.
    a, b, c are [x, y] coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def draw_landmarks_and_connections(image, landmarks):
    """
    Manually draws landmarks and connections on the image.
    image: BGR numpy array
    landmarks: list of NormalizedLandmark objects
    """
    h, w, c = image.shape
    
    # Draw Connections
    for start_idx, end_idx in CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(image, start_point, end_point, (245, 117, 66), 2)

    # Draw Points
    for i, lm in enumerate(landmarks):
        # Only draw relevant landmarks (Body) to keep it clean, or all.
        if i > 10 and i < 29: # Skip face (0-10) and feet/hands detail
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (245, 66, 230), -1)

class PoseEstimator(VideoTransformerBase):
    def __init__(self):
        # Initialize PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to MediaPipe Image
        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Calculate timestamp (monotonic)
        # Using a simple counter or system time. 
        # Video mode requires strictly increasing timestamps.
        self.timestamp_ms += 100 # Approx 10fps increment, or use time.time
        # Better to use actual time derived from frame if available, but manual increment follows sequence
        current_timestamp = int(time.time() * 1000)
        
        try:
            detection_result = self.landmarker.detect_for_video(mp_image, current_timestamp)
            
            # Draw results
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0] # First pose
                
                # Draw skeleton
                draw_landmarks_and_connections(img, landmarks)
                
                # Analyze Squat (Left Leg)
                # Indices: 23 (Hip), 25 (Knee), 27 (Ankle)
                hip = [landmarks[23].x, landmarks[23].y]
                knee = [landmarks[25].x, landmarks[25].y]
                ankle = [landmarks[27].x, landmarks[27].y]
                
                angle = calculate_angle(hip, knee, ankle)
                
                # Status Logic
                status = "Standing"
                color = (0, 255, 255) # Yellow
                
                if angle > 160:
                    status = "Standing"
                    color = (0, 255, 255)
                elif angle < 140 and angle > 70:
                    status = "Good Squat"
                    color = (0, 255, 0) # Green
                elif angle < 70:
                    status = "Too Deep"
                    color = (0, 0, 255) # Red
                
                # Draw Text
                h, w, c = img.shape
                knee_pos = (int(knee[0] * w), int(knee[1] * h))
                
                cv2.putText(img, str(int(angle)), knee_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Info Box
                cv2.rectangle(img, (0,0), (250,80), (245,117,16), -1)
                cv2.putText(img, status, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(img, f"Angle: {int(angle)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                
        except Exception as e:
            print(e)
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🏋️ AI Exercise Form Corrector (Live)")
st.write("Powered by MediaPipe Tasks (Python 3.13 Compatible)")

st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Mode", ["Live AI Trainer", "Upload Image"])

if mode == "Live AI Trainer":
    st.write("## Live Squat Analysis")
    st.write("Click 'Start'. Stand back so your full body is visible.")
    
    ctx = webrtc_streamer(key="squat-counter", video_transformer_factory=PoseEstimator)

elif mode == "Upload Image":
    st.write("## Image Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # For image mode, we initialize a separate detector in IMAGE mode
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.pose_landmarks:
                # Draw on a copy
                annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                landmarks = detection_result.pose_landmarks[0]
                
                draw_landmarks_and_connections(annotated_image, landmarks)
                
                # Calc Angle logic (same as above)
                hip = [landmarks[23].x, landmarks[23].y]
                knee = [landmarks[25].x, landmarks[25].y]
                ankle = [landmarks[27].x, landmarks[27].y]
                angle = calculate_angle(hip, knee, ankle)
                
                st.write(f"### Detected Knee Angle: **{int(angle)}°**")
                
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image, caption="Analyzed Result")
            else:
                st.error("No person detected.")
