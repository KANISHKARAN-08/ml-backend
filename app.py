import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU only (important on Render)

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import savgol_filter
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()

# ----------------------------
# Load MoveNet Lightning (fast + small)
# ----------------------------
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def detect_pose(image):
    """Run MoveNet on a single frame"""
    img = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)  # 192x192 lighter than 256x256
    keypoints = movenet.signatures['serving_default'](tf.cast(img, dtype=tf.int32))
    return keypoints['output_0'].numpy()[0, 0, :, :2]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ab, cb = a - b, c - b
    dot = np.dot(ab, cb)
    norm = np.linalg.norm(ab) * np.linalg.norm(cb)
    if norm == 0:
        return np.nan
    return np.degrees(np.arccos(dot / norm))

def smooth_series(series):
    if len(series) < 7:
        return series
    return savgol_filter(series, window_length=7, polyorder=2)

def detect_reps(angles, down_thresh, up_thresh, exercise_type="pushups"):
    phase, reps, correct, incorrect, suggestions = "up", 0, 0, 0, []
    for ang in angles:
        if np.isnan(ang):
            continue
        if ang < down_thresh and phase == "up":
            phase = "down"
        elif ang > up_thresh and phase == "down":
            phase = "up"
            reps += 1
            if down_thresh - 10 < ang < up_thresh + 10:
                correct += 1
            else:
                incorrect += 1
                # Dynamic suggestions
                if exercise_type == "pushups":
                    if ang < down_thresh:
                        suggestions.append("Go lower during pushup.")
                    elif ang > up_thresh:
                        suggestions.append("Lock your elbows fully at the top.")
                    else:
                        suggestions.append("Maintain proper pushup form.")
                elif exercise_type == "squats":
                    if ang < down_thresh:
                        suggestions.append("Squat deeper for full range.")
                    elif ang > up_thresh:
                        suggestions.append("Stand fully upright at the top.")
                    else:
                        suggestions.append("Keep knees aligned with toes.")
    return reps, correct, incorrect, suggestions

# ----------------------------
# Exercise analyzers
# ----------------------------
def analyze_pushups(video_path):
    cap = cv2.VideoCapture(video_path)
    angles, frame_count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 5 != 0:  # skip frames to save memory
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = detect_pose(frame)
        angle = calculate_angle(keypoints[6], keypoints[8], keypoints[10])  # arm
        angles.append(angle)
    cap.release()
    angles = smooth_series(np.array(angles))
    reps, correct, incorrect, suggestions = detect_reps(
        angles, down_thresh=100, up_thresh=150, exercise_type="pushups"
    )
    return {
        "exercise": "pushups",
        "total_reps": reps,
        "correct_reps": correct,
        "incorrect_reps": incorrect,
        "overall_score": correct * 10,
        "suggestions": suggestions
    }

def analyze_squats(video_path):
    cap = cv2.VideoCapture(video_path)
    angles, frame_count = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 5 != 0:  # skip frames to save memory
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = detect_pose(frame)
        angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])  # leg
        angles.append(angle)
    cap.release()
    angles = smooth_series(np.array(angles))
    reps, correct, incorrect, suggestions = detect_reps(
        angles, down_thresh=100, up_thresh=160, exercise_type="squats"
    )
    return {
        "exercise": "squats",
        "total_reps": reps,
        "correct_reps": correct,
        "incorrect_reps": incorrect,
        "overall_score": correct * 10,
        "suggestions": suggestions
    }

# ----------------------------
# API Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "Sports Talent ML API Running!"}

@app.post("/analyze")
async def analyze_video(exercise_type: str = Form(...), video: UploadFile = None):
    try:
        temp_file = f"temp_{video.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        if exercise_type.lower() == "pushups":
            result = analyze_pushups(temp_file)
        elif exercise_type.lower() == "squats":
            result = analyze_squats(temp_file)
        else:
            result = {"error": "Unsupported exercise type"}

        os.remove(temp_file)
        return result

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
