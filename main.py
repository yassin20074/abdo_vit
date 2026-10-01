import os
import asyncio
import httpx
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import User3DResponse, FaceAnalysis, Landmark3D

app = FastAPI(
    title="User 3D Model & Face Landmarks API",
    description="API to generate 3D model from user photo and extract 3D face mesh coordinates",
    version="1.0.0"
)

# السماح لجميع المصادر بالوصول (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قراءة مفتاح الـ API من متغيرات البيئة في Railway
TRIPO_API_KEY = os.getenv("TRIPO_API_KEY")
TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"

# تهيئة MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def extract_face_landmarks(image_bytes: bytes) -> FaceAnalysis:
    """تحليل صورة الشخص واستخراج جميع إحداثيات نقاط الوجه (468+ نقطة)"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="الصورة المرسلة غير صالحة")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return FaceAnalysis(detected=False, landmarks=[])

    face_landmarks = results.multi_face_landmarks[0]
    landmarks_list = [
        Landmark3D(id=idx, x=lm.x, y=lm.y, z=lm.z)
        for idx, lm in enumerate(face_landmarks.landmark)
    ]

    return FaceAnalysis(detected=True, landmarks=landmarks_list)


async def generate_3d_from_tripo(image_bytes: bytes, filename: str) -> dict:
    """رفع صورة الشخص لـ Tripo3D وإرجاع رابط ملف الـ 3D"""
    if not TRIPO_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="لم يتم ضبط TRIPO_API_KEY في متغيرات البيئة"
        )

    headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. رفع صورة الشخص للحصول على image_token
        files = {"file": (filename, image_bytes, "image/jpeg")}
        upload_res = await client.post(f"{TRIPO_BASE_URL}/upload", headers=headers, files=files)
        
        if upload_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"فشل رفع الصورة: {upload_res.text}")

        image_token = upload_res.json().get("data", {}).get("image_token")

        # 2. بدء مهمة تحويل الصورة إلى مجسم 3D
        task_payload = {
            "type": "image_to_3d",
            "file": {
                "type": "jpg",
                "file_token": image_token
            }
        }
        task_res = await client.post(f"{TRIPO_BASE_URL}/task", headers=headers, json=task_payload)
        
        if task_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"فشل بدء عملية الـ 3D: {task_res.text}")

        task_id = task_res.json().get("data", {}).get("task_id")

        # 3. متابعة حالة المهمة حتى تجهيز ملف الـ 3D (GLB)
        model_url = None
        for _ in range(30):  # محاولة الاستعلام لمدة تصل لـ 60 ثانية
            await asyncio.sleep(2)
            status_res = await client.get(f"{TRIPO_BASE_URL}/task/{task_id}", headers=headers)
            if status_res.status_code == 200:
                res_data = status_res.json().get("data", {})
                current_status = res_data.get("status")

                if current_status == "success":
                    output = res_data.get("output", {})
                    # جلب رابط ملف الـ 3D
                    model_url = output.get("model") or output.get("pbr_model")
                    break
                elif current_status in ["failed", "cancelled"]:
                    raise HTTPException(status_code=500, detail="فشلت عملية توليد ملف الـ 3D من الصورة")
                  return {"task_id": task_id, "model_url": model_url}


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}


@app.post("/api/v1/process-user-image", response_model=User3DResponse)
async def process_user_image(file: UploadFile = File(...)):
    """
    استقبال صورة الشخص -> 
    إرجاع إحداثيات الوجه الـ 3D + رابط ملف المجسم 3D الخاص بالشخص
    """
    image_bytes = await file.read()

    # 1. استخراج إحداثيات وجه الشخص
    face_landmarks = extract_face_landmarks(image_bytes)

    # 2. توليد ملف الـ 3D للشخص عبر Tripo3D
    tripo_result = await generate_3d_from_tripo(image_bytes, file.filename)

    return User3DResponse(
        status="success",
        tripo_task_id=tripo_result["task_id"],
        tripo_model_url=tripo_result["model_url"],
        face_landmarks=face_landmarks
    )
