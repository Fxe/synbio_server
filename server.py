from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uuid
from typing import Dict
import io
from fastapi.middleware.cors import CORSMiddleware
from service import SynbioService


description = """
SynBio API
"""

app = FastAPI(
    title="SynBio",
    description=description,
    summary="API",
    version="0.0.1",
)

# Store experiments in memory for simplicity
EXPERIMENTS: Dict[str, Dict] = {}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/whoiam")
def get_whoiam():
    return JSONResponse(content="SynBio Server")


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

    service = SynbioService(None, None)
    #service.etl()

    return {"experiment_id": experiment_id}


@app.post("/lab")
async def add_lab(some_lab_parm: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.post("/strain")
async def add_strain(some_param: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/strain")
async def list_strains():
    # get strains
    # call database service
    # list all strains
    # return
    return JSONResponse(content=list(["strain1", "strain2"]))


@app.get("/experiment/{experiment_id}/{condition}")
def get_experiment_condition(experiment_id: str, condition: str):
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")

    df = EXPERIMENTS[experiment_id]["data"]

    if condition not in df.columns:
        raise HTTPException(status_code=400, detail=f"Condition '{condition}' not found in data")

    result_df = df[[condition]]

    return JSONResponse(content=result_df.to_dict(orient="records"))


@app.get("/experiment")
def list_experiments():
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/experiment/{experiment_id}")
def list_experiment_conditions(experiment_id: str):
    if experiment_id not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return JSONResponse(content=list(EXPERIMENTS))
