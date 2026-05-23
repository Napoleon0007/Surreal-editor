FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir opencv-python-headless
WORKDIR /app
COPY . .
CMD ["python3", "app.py"]
