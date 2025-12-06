# ============================
# 1. Base image
# ============================
FROM python:3.12-slim

# Prevent Python from creating .pyc files & enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================
# 2. Set working directory
# ============================
WORKDIR /app

# ============================
# 3. Install system dependencies
#    (LightGBM & XGBoost often need these)
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 4. Copy project files
# ============================
COPY . .

# ============================
# 5. Install Python dependencies
# ============================
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# 6. Expose FastAPI port
# ============================
EXPOSE 8000

# ============================
# 7. Start FastAPI app
# ============================
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
