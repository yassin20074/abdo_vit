import logging
import os
import math
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("virtual-tryon-api")

app = FastAPI(title="Virtual Try-On Face Landmark API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "face_landmarker.task"

# تنزيل نموذج MediaPipe تلقائياً إن لم يكن موجوداً
if not os.path.exists(MODEL_PATH):
    import urllib.request
    try:
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        logger.info("Downloading face_landmarker.task...")
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        logger.info("Download completed.")
    except Exception as e:
        logger.error(f"Failed to download MediaPipe model: {str(e)}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Pydantic Schemas لإرجاع البيانات المطلوبة للـ Frontend (Three.js)
class GlassesTransform(BaseModel):
    detected: bool
    position: List[float]  # [x, y, z] - مركز النظارة على الوجه
    scale: List[float]     # [scale_x, scale_y, scale_z] - حجم النظارة بناءً على عرض الوجه والمسافة بين العينين
    rotation: List[float]  # [pitch, yaw, roll] - زوايا دوران الرأس لتوجيه النظارة بميل صحيح

class FaceAnalysisResponse(BaseModel):
    status: str
    glasses_transform: GlassesTransform


def calculate_glasses_transform(image_bytes: bytes) -> GlassesTransform:
    """استخراج مصفوفات التحويل (Position, Scale, Rotation) بدقة عالية من صورة المستخدم 2D"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return GlassesTransform(detected=False, position=[0,0,0], scale=[1,1,1], rotation=[0,0,0])

        h_orig, w_orig, _ = img_bgr.shape

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = landmarker.detect(mp_image)

            crop_x_off, crop_y_off = 0, 0
            curr_w, curr_h = w_orig, h_orig

            # Adaptive Crop fallback في حال كانت الصورة بعيدة أو تحتاج تحسين كشف الوجه
            if not detection_result.face_landmarks:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

                if len(faces) > 0:
                    x, y, w_box, h_box = max(faces, key=lambda rect: rect[2] * rect[3])
                    margin = int(max(w_box, h_box) * 0.8)
                    x1, y1 = max(0, x - margin), max(0, y - margin)
                    x2, y2 = min(w_orig, x + w_box + margin), min(h_orig, y + h_box + margin)
                    
                    cropped_face = img_bgr[y1:y2, x1:x2]
                    crop_x_off, crop_y_off = x1, y1
                    curr_h, curr_w, _ = cropped_face.shape
                    crop_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                    crop_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                    detection_result = landmarker.detect(crop_mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]

                # نقاط أساسية من MediaPipe لاستخراج القياسات
                l_eye = landmarks[33]   # العين اليسرى (الخارجية)
                r_eye = landmarks[263]  # العين اليمنى (الخارجية)
                l_temple = landmarks[127] # الصدغ الأيسر
                r_temple = landmarks[356] # الصدغ الأيمن
                nose_bridge = landmarks[168] # جسر الأنف العلوي
                chin = landmarks[152]   # الذقن

                # تحويل النونورمالايزد إلى بكسلات حقيقية
                lx, ly, lz = l_eye.x * curr_w + crop_x_off, l_eye.y * curr_h + crop_y_off, l_eye.z * curr_w
                rx, ry, rz = r_eye.x * curr_w + crop_x_off, r_eye.y * curr_h + crop_y_off, r_eye.z * curr_w
                nx, ny, nz = nose_bridge.x * curr_w + crop_x_off, nose_bridge.y * curr_h + crop_y_off, nose_bridge.z * curr_w
                lt_x, lt_y = l_temple.x * curr_w + crop_x_off, l_temple.y * curr_h + crop_y_off
                rt_x, rt_y = r_temple.x * curr_w + crop_x_off, r_temple.y * curr_h + crop_y_off
                chin_y = chin.y * curr_h + crop_y_off

                # 1. POSITION [x, y, z] - مركز النظارة (مستند لجسر الأنف والعيون)
                pos_x = round(float(nx), 2)
                pos_y = round(float(ny), 2)
                pos_z = round(float(nz), 2)

                # 2. SCALE [scale_x, scale_y, scale_z] - أبعاد النظارة وعمقها بناءً على عرض الوجه والمسافة بين العينين
                face_width = math.hypot(rt_x - lt_x, rt_y - lt_y)
                eye_dist = math.hypot(rx - lx, ry - ly)
                
                scale_x = round(float(face_width), 2)
                scale_y = round(float(scale_x * 0.45), 2) # النسبة الطبيعية لارتفاع النظارة
                scale_z = round(float(eye_dist * 0.5), 2)   # عمق ذراع النظارة

                # 3. ROTATION [pitch, yaw, roll] بالدرجات لضبط ميل النظارة مع حركة رأس المستخدم
                dx = rx - lx
                dy = ry - ly
                roll = round(float(math.degrees(math.atan2(dy, dx))), 2)

                dz = rz - lz
                yaw = round(float(math.degrees(math.atan2(dz, dx))), 2)

                dy_pitch = chin_y - ny
                dz_pitch = (chin.z * curr_w) - nz
                pitch = round(float(math.degrees(math.atan2(dz_pitch, dy_pitch))), 2)

                return GlassesTransform(
                    detected=True,
                    position=[pos_x, pos_y, pos_z],
                    scale=[scale_x, scale_y, scale_z],
                    rotation=[pitch, yaw, roll]
                )

        return GlassesTransform(detected=False, position=[0,0,0], scale=[1,1,1], rotation=[0,0,0])

    except Exception as e:
        logger.error(f"Error extracting glasses transform: {str(e)}")
        return GlassesTransform(detected=False, position=[0,0,0], scale=[1,1,1], rotation=[0,0,0])


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Virtual Try-On Landmark API (Zero-Cost Architecture)"}


@app.post("/api/v1/analyze-face", response_model=FaceAnalysisResponse)
async def analyze_face(file: UploadFile = File(...)):

    image_bytes = await file.read()
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded")

    glasses_transform = calculate_glasses_transform(image_bytes)

    return FaceAnalysisResponse(
        status="success",
        glasses_transform=glasses_transform
    )
