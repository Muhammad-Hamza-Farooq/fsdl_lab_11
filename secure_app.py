import logging
import time
from collections import defaultdict
from typing import List, Union

from fastapi import FastAPI, HTTPException, Request

security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/security.log", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
security_logger.addHandler(handler)

app = FastAPI(title="Secure ML API")
rate_limit_store = defaultdict(list)
RATE_LIMIT = 5
WINDOW_SECONDS = 10


def check_rate_limit(client_ip: str):
    now = time.time()
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < WINDOW_SECONDS]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        security_logger.warning("Rate limit exceeded for %s", client_ip)
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rate_limit_store[client_ip].append(now)


def sanitize(data: List[Union[int, float]]):
    if len(data) > 1000:
        security_logger.warning("Huge payload blocked: size=%s", len(data))
        raise HTTPException(status_code=413, detail="Payload too large")
    if not all(isinstance(x, (int, float)) and abs(x) < 1e6 for x in data):
        security_logger.warning("Invalid/adversarial input blocked")
        raise HTTPException(status_code=422, detail="Invalid input")
    return data


@app.post("/predict")
def predict(request: Request, data: List[Union[int, float]]):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit(client_ip)
    security_logger.info("Predict request from %s", client_ip)

    clean = sanitize(data)
    return {"prediction": sum(clean), "version": "v1"}
