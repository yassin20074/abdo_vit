import logging
import os
import asyncio
import httpx
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="User 3D Model & Face Landmarks API", version="1.0.0")

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

# تنزيل نموذج MediaPipe التلقائي إن لم يكن موجوداً
if not os.path.exists(MODEL_PATH):
    import urllib.request
    try:
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        logger.info("Downloading face_landmarker.task...")
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        logger.info("Download completed.")
    except Exception as e:
        logger.error(f"Failed to download MediaPipe model: {str(e)}")

# Pydantic Schemas للـ Response المطابق لطلبك
class GlassesLandmarker(BaseModel):
    detected: bool
    center_x: int
    center_y: int
    glasses_width: int
    angle: float

class User3DResponse(BaseModel):
    status: str
    id: str
    tripo_model_url: Optional[str] = None
    landmarker: GlassesLandmarker


def extract_glasses_landmarker(image_bytes: bytes) -> GlassesLandmarker:
    """استخراج حسابات النظارة (الموقع، العرض، الزاوية) بالبكسل"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Failed to decode image with OpenCV")
            return GlassesLandmarker(detected=False, center_x=0, center_y=0, glasses_width=0, angle=0.0)

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                logger.warning("No face detected in image")
                return GlassesLandmarker(detected=False, center_x=0, center_y=0, glasses_width=0, angle=0.0)

            landmarks = detection_result.face_landmarks[0]
            
            # النقاط المرجعية:
            # 33: العين اليسرى الخارجية | 263: العين اليمنى الخارجية
            # 127: الصدغ الأيسر | 356: الصدغ الأيمن
            l_eye = landmarks[33]
            r_eye = landmarks[263]
            l_temple = landmarks[127]
            r_temple = landmarks[356]

            # تحويل الإحداثيات إلى Pixels
            lx, ly = l_eye.x * w, l_eye.y * h
            rx, ry = r_eye.x * w, r_eye.y * h
            lt_x, lt_y = l_temple.x * w, l_temple.y * h
            rt_x, rt_y = r_temple.x * w, r_temple.y * h

            # 1. منتصف النظارة (Center X, Center Y)
            center_x = int(round((lx + rx) / 2))
            center_y = int(round((ly + ry) / 2))

            # 2. عرض النظارة بناءً على المسافة بين الصدغين
            glasses_width = int(round(np.sqrt((rt_x - lt_x)**2 + (rt_y - lt_y)**2)))

            # 3. زاوية ميلان الوجه (Angle in Degrees)
            dx = rx - lx
            dy = ry - ly
            angle_rad = np.arctan2(dy, dx)
            angle_deg = round(float(np.degrees(angle_rad)), 2)
            return GlassesLandmarker(
                detected=True,
                center_x=center_x,
                center_y=center_y,
                glasses_width=glasses_width,
                angle=angle_deg
            )
    except Exception as e:
        logger.error(f"Error in extract_glasses_landmarker: {str(e)}")
        return GlassesLandmarker(detected=False, center_x=0, center_y=0, glasses_width=0, angle=0.0)


async def generate_3d_from_tripo(image_bytes: bytes, filename: str) -> dict:
    if not TRIPO_API_KEY:
        logger.error("TRIPO_API_KEY is missing!")
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {TRIPO_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            # 1. رفع الصورة
            files = {"file": (filename, image_bytes, "image/jpeg")}
            upload_headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
            
            upload_res = await client.post(f"{TRIPO_BASE_URL}/files", headers=upload_headers, files=files)
            if upload_res.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Tripo upload failed: {upload_res.text}")

            upload_data = upload_res.json().get("data", {})
            file_token = upload_data.get("file_token") or upload_data.get("image_token")

            # 2. إنشاء مهمة التوليد مع الحفاظ على ملامح الصورة الحقيقية (Original Image Texture Alignment)
            task_payload = {
                "model": "v3.0-20250812",
                "file": {
                    "type": "jpg",
                    "file_token": file_token
                },
                "texture_alignment": "original_image",  # مطابقة الخامات مع الصورة الأصلية للحفاظ على الملامح
                "texture_quality": "detailed",          # دقة خامات عالية
                "pbr": True                             # تفعيل PBR لواقعية الإضاءة
            }

            task_res = await client.post(f"{TRIPO_BASE_URL}/generation/image-to-model", headers=headers, json=task_payload)
            if task_res.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Tripo task creation failed: {task_res.text}")

            task_id = task_res.json().get("data", {}).get("task_id")

            # 3. Polling ممتد حتى تجهيز المجسم
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
                        logger.error(f"Tripo Task Failed Status: {res_data}")
                        raise HTTPException(status_code=500, detail="Tripo 3D generation task failed")

            return {"task_id": task_id, "model_url": model_url}
    except Exception as e:
        logger.error(f"Error in generate_3d_from_tripo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tripo3D API Error: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}


@app.post("/api/v1/process-user-image", response_model=User3DResponse)
async def process_user_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # استخراج الحسابات
    landmarker_data = extract_glasses_landmarker(image_bytes)
    
    # توليد المجسم 3D
    tripo_result = await generate_3d_from_tripo(image_bytes, file.filename)
    return User3DResponse(
        status="success",
        id=tripo_result["task_id"],
        tripo_model_url=tripo_result["model_url"],
        landmarker=landmarker_data
    )
