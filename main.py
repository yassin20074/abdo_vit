import logging
import os
import asyncio
import math
import httpx
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("virtual-tryon-api")

app = FastAPI(title="3D Virtual Try-On API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")
MODEL_PATH = "face_landmarker.task"
TRIPO_BASE_URL = "https://openapi.tripo3d.ai/v3"

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

# Pydantic Schemas مخصصة للتحول والتركيب الـ 3D
class GlassesTransform(BaseModel):
    detected: bool
    position: List[float]  # [x, y, z] - مركز النظارة في الصورة/الفضاء
    scale: List[float]     # [scale_x, scale_y, scale_z] - الحجم والسكيل
    rotation: List[float]  # [pitch, yaw, roll] - زوايا الدوران بالدرجات

class User3DResponse(BaseModel):
    status: str
    task_id: str
    tripo_model_url: Optional[str] = None
    glasses_transform: GlassesTransform


def calculate_glasses_transform(image_bytes: bytes) -> GlassesTransform:
    """استخراج مصفوفات التحويل (Position, Scale, Rotation) للنظارة من الصورة الأمامية"""
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
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = landmarker.detect(mp_image)

            crop_x_off, crop_y_off = 0, 0
            curr_w, curr_h = w_orig, h_orig

            # Adaptive Crop fallback للوجوه البعيدة
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

                # Points key:
                # 33: Left eye outer, 263: Right eye outer
                # 127: Left temple, 356: Right temple
                # 6: Nose bridge top, 1: Nose tip
                # 10: Forehead top, 152: Chin bottom
                l_eye = landmarks[33]
                r_eye = landmarks[263]
                l_temple = landmarks[127]
                r_temple = landmarks[356]
                nose_bridge = landmarks[168]
                chin = landmarks[152]

                # Denormalize to pixel values
                lx, ly, lz = l_eye.x * curr_w + crop_x_off, l_eye.y * curr_h + crop_y_off, l_eye.z * curr_w
                rx, ry, rz = r_eye.x * curr_w + crop_x_off, r_eye.y * curr_h + crop_y_off, r_eye.z * curr_w
                nx, ny, nz = nose_bridge.x * curr_w + crop_x_off, nose_bridge.y * curr_h + crop_y_off, nose_bridge.z * curr_w
                lt_x, lt_y = l_temple.x * curr_w + crop_x_off, l_temple.y * curr_h + crop_y_off
                rt_x, rt_y = r_temple.x * curr_w + crop_x_off, r_temple.y * curr_h + crop_y_off
                chin_y = chin.y * curr_h + crop_y_off

                # 1. POSITION [x, y, z]
                pos_x = round(float(nx), 2)
                pos_y = round(float(ny), 2)
                pos_z = round(float(nz), 2)  # depth relative offset

                # 2. SCALE [scale_x, scale_y, scale_z]
                # عرض النظارة المقترح بناءً على المسافة بين الصدغين
                face_width = math.hypot(rt_x - lt_x, rt_y - lt_y)
                eye_dist = math.hypot(rx - lx, ry - ly)
                
                scale_x = round(float(face_width), 2)
                scale_y = round(float(scale_x * 0.45), 2) # نسبة الارتفاع الافتراضية للنظارة
                scale_z = round(float(eye_dist * 0.5), 2)   # عمق الذراع الافتراضي للنظارة

                # 3. ROTATION [pitch, yaw, roll] in Degrees
                # Roll (Z-axis rotation): دوران الوجه يميناً ويساراً في المستوى
                dx = rx - lx
                dy = ry - ly
                roll = round(float(math.degrees(math.atan2(dy, dx))), 2)

                # Yaw (Y-axis rotation): التفات الوجه يميناً ويساراً
                dz = rz - lz
                yaw = round(float(math.degrees(math.atan2(dz, dx))), 2)

                # Pitch (X-axis rotation): إمالة الوجه لأعلى ولأسفل
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


async def upload_file_to_tripo(client: httpx.AsyncClient, file_bytes: bytes, filename: str) -> str:
    """رفع صورة مفردة إلى Tripo والحصول على file_token"""
    files = {"file": (filename, file_bytes, "image/jpeg")}
    upload_headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
    
    res = await client.post(f"{TRIPO_BASE_URL}/files", headers=upload_headers, files=files)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to upload {filename} to Tripo: {res.text}")
    
    data = res.json().get("data", {})
    return data.get("file_token") or data.get("image_token")

async def generate_multiview_3d_tripo(front_bytes: bytes, right_bytes: bytes, left_bytes: bytes) -> dict:
    """إرسال الـ 3 صور إلى Tripo Multi-View API مع الهيكل الصحيح للـ Payload"""
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY environment variable is missing")

    headers = {
        "Authorization": f"Bearer {TRIPO_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        # 1. رفع الصور الثلاث والحصول على الـ tokens
        front_token, right_token, left_token = await asyncio.gather(
            upload_file_to_tripo(client, front_bytes, "front.jpg"),
            upload_file_to_tripo(client, right_bytes, "right.jpg"),
            upload_file_to_tripo(client, left_bytes, "left.jpg")
        )

        # 2. الهيكل الصحيح لـ Multi-View Generation في Tripo3D v3
        task_payload = {
            "type": "multiview_to_model",
            "files": [
                {"type": "jpg", "file_token": front_token},
                {"type": "jpg", "file_token": right_token},
                {"type": "jpg", "file_token": left_token}
            ],
            "model": "v3.0-20250812",
            "texture_alignment": "original_image",
            "texture_quality": "detailed",
            "pbr": True
        }

       

        task_res = await client.post(f"{TRIPO_BASE_URL}/generation/image-to-model", headers=headers, json=task_payload)
        
        if task_res.status_code != 200:
            logger.error(f"Tripo Error Response: {task_res.text}")
            raise HTTPException(status_code=500, detail=f"Tripo Multi-View Task Creation Failed: {task_res.text}")

        task_id = task_res.json().get("data", {}).get("task_id")

        # 3. Polling لمتابعة تجهيز المجسم
        model_url = None
        for _ in range(75):
            await asyncio.sleep(2)
            status_res = await client.get(f"{TRIPO_BASE_URL}/tasks/{task_id}", headers=headers)
            
            if status_res.status_code == 200:
                res_data = status_res.json().get("data", {})
                current_status = res_data.get("status")
                
                if current_status == "success":
                    output = res_data.get("output", {})
                    model_url = output.get("model_url") or output.get("pbr_model_url") or output.get("model")
                    break
                elif current_status in ["failed", "cancelled", "banned"]:
                    logger.error(f"Tripo Task Failed: {res_data}")
                    raise HTTPException(status_code=500, detail="Tripo 3D generation task failed")

        return {"task_id": task_id, "model_url": model_url}


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Virtual Try-On 3D Processing Engine"}


@app.post("/api/v1/process-user-multiview", response_model=User3DResponse)
async def process_user_multiview(
    front_file: UploadFile = File(...),
    right_file: UploadFile = File(...),
    left_file: UploadFile = File(...)
):
    # قراءة بكسلات الصور الثلاثة
    front_bytes = await front_file.read()
    right_bytes = await right_file.read()
    left_bytes = await left_file.read()

    # 1. استخراج إحداثيات النظارة (Position, Scale, Rotation) من الصورة الأمامية
    glasses_transform = calculate_glasses_transform(front_bytes)

    # 2. توليد المجسم 3D باستخدام Multi-View عبر Tripo3D
    tripo_result = await generate_multiview_3d_tripo(front_bytes, right_bytes, left_bytes)

    return User3DResponse(
        status="success",
        task_id=tripo_result["task_id"],
        tripo_model_url=tripo_result["model_url"],
        glasses_transform=glasses_transform
    )
