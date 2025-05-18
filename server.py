from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uuid
from typing import Dict
import io

app = FastAPI()

# Store experiments in memory for simplicity
EXPERIMENTS: Dict[str, Dict] = {}

@app.post("/experiment/upload")
async def upload_experiment(
    file: UploadFile = File(...),
    experiment_name: str = Form(...),
    description: str = Form(None)
):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), sep=",")  # assuming tab-separated text file
        print(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {e}")

    experiment_id = str(uuid.uuid4())
    EXPERIMENTS[experiment_id] = {
        "name": experiment_name,
        "description": description,
        "data": df
    }

    return {"experiment_id": experiment_id}

@app.get("/experiment/{experiment_id}/{condition}")
def get_experiment_condition(experiment_id: str, condition: str):
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")

    df = EXPERIMENTS[experiment_id]["data"]

    if condition not in df.columns:
        raise HTTPException(status_code=400, detail=f"Condition '{condition}' not found in data")

    result_df = df[[condition]]

    return JSONResponse(content=result_df.to_dict(orient="records"))
