FROM python:3.11

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements-docker.txt

COPY . .

# Default: run training script (lab)
CMD ["python", "training/train.py"]
