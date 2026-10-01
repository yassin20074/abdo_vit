import logging
import os
import asyncio
import httpx
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import User3DResponse, FaceAnalysis, Landmark3D

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

# التحديث الأهم: استخدام إصدار V3 API الرسمية
TRIPO_BASE_URL = "https://api.tripo3d.ai/v3/openapi"

MODEL_PATH = "face_landmarker.task"

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


def extract_face_landmarks(image_bytes: bytes) -> FaceAnalysis:
    """استخراج معالم الوجه الـ 3D باستخدام MediaPipe Tasks API"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="الصورة المرسلة غير صالحة")

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
            min_face_detection_confidence=0.5
        )

        with FaceLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)

            if not detection_result.face_landmarks:
                return FaceAnalysis(detected=False, landmarks=[])

            face_landmarks = detection_result.face_landmarks[0]
            landmarks_list = [
                Landmark3D(id=idx, x=lm.x, y=lm.y, z=lm.z)
                for idx, lm in enumerate(face_landmarks)
            ]

            return FaceAnalysis(detected=True, landmarks=landmarks_list)
    except Exception as e:
        logger.error(f"Error in extract_face_landmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MediaPipe processing error: {str(e)}")


async def generate_3d_from_tripo(image_bytes: bytes, filename: str) -> dict:
    """رفع الصورة وتوليد مجسم 3D عبر Tripo3D V3 API"""
    if not TRIPO_API_KEY:
        logger.error("TRIPO_API_KEY is missing in environment variables!")
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {TRIPO_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. رفع الصورة للحصول على file_token
            files = {"file": (filename, image_bytes, "image/jpeg")}
            upload_headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
            upload_res = await client.post(f"{TRIPO_BASE_URL}/upload", headers=upload_headers, files=files)
            
            if upload_res.status_code != 200:
                logger.error(f"Tripo Upload Failed: {upload_res.text}")
                raise HTTPException(status_code=500, detail=f"Tripo upload failed: {upload_res.text}")
            image_token = upload_res.json().get("data", {}).get("image_token")

            # 2. إنشاء مهمة توليد الـ 3D بتنسيق V3 API
            task_payload = {
                "type": "image_to_model",
                "model_version": "v3.0",
                "file": {
                    "type": "jpg",
                    "file_token": image_token
                }
            }

            task_res = await client.post(f"{TRIPO_BASE_URL}/task", headers=headers, json=task_payload)
            
            if task_res.status_code != 200:
                logger.error(f"Tripo Task Creation Failed: {task_res.text}")
                raise HTTPException(status_code=500, detail=f"Tripo task creation failed: {task_res.text}")

            task_id = task_res.json().get("data", {}).get("task_id")

            # 3. متابعة حالة التوليد (Polling Loop)
            model_url = None
            for _ in range(30):
                await asyncio.sleep(2)
                status_res = await client.get(f"{TRIPO_BASE_URL}/task/{task_id}", headers=headers)
                if status_res.status_code == 200:
                    res_data = status_res.json().get("data", {})
                    current_status = res_data.get("status")

                    if current_status == "success":
                        output = res_data.get("output", {})
                        model_url = output.get("model") or output.get("pbr_model")
                        break
                    elif current_status in ["failed", "cancelled"]:
                        logger.error(f"Tripo Task Failed: {res_data}")
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
    face_landmarks = extract_face_landmarks(image_bytes)
    tripo_result = await generate_3d_from_tripo(image_bytes, file.filename)

    return User3DResponse(
        status="success",
        tripo_task_id=tripo_result["task_id"],
        tripo_model_url=tripo_result["model_url"],
        face_landmarks=face_landmarks
    )
