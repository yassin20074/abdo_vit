from pydantic import BaseModel, Field
from typing import List, Optional

class Landmark3D(BaseModel):
    id: int
    x: float = Field(..., description="احداثي X الموزون بين 0 و 1")
    y: float = Field(..., description="احداثي Y الموزون بين 0 و 1")
    z: float = Field(..., description="احداثي العمق Z")

class FaceAnalysis(BaseModel):
    detected: bool
    landmarks: List[Landmark3D] = []

class User3DResponse(BaseModel):
    status: str
    tripo_task_id: str
    tripo_model_url: Optional[str] = Field(None, description="رابط ملف الـ 3D (GLB) الخص بالمستخدم المولد من Tripo3D")
    face_landmarks: FaceAnalysis
