FROM python:3.12.3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


# dependencias del sistema para opencv (si usas mtcnn/opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
libglib2.0-0 libsm6 libxrender1 libxext6 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./app/
COPY models/ ./models/


ENV MODEL_PATH="/app/models/attitude_model.keras"


EXPOSE 8000


CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]