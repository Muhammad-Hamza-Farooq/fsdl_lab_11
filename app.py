from typing import List, Union

from fastapi import FastAPI, HTTPException

app = FastAPI()


@app.post("/predict")
def predict(data: List[Union[int, float]]):
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty payload")

    if len(data) > 10_000:
        raise HTTPException(status_code=413, detail="Payload too large")

    return {"prediction": sum(data)}
