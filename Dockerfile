# استخدام نسخة Python خفيفة ومستقرة
FROM python:3.10-slim

# ضبط بيئة العمل داخل الـ Container
WORKDIR /app

# منع Python من كتابة ملفات pyc وضبط المخرجات لتظهر مباشرة في الـ Logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# تثبيت المكتبات البرمجية الخاصة بالنظام والمطلوبة لعمل OpenCV و MediaPipe بدون مشاكل
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات أولاً للاستفادة من Docker Cache
COPY requirements.txt .

# تثبيت مكتبات Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى الـ Container
COPY schemas.py .
COPY main.py .
# فتح المنفذ (Port) الخاص بالتطبيق
EXPOSE 8080

# تشغيل السيرفر باستخدام Uvicorn على المنفذ المحدد لـ Railway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
