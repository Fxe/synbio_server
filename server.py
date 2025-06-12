from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import uuid
from typing import Dict
import io
from fastapi.middleware.cors import CORSMiddleware
from service import SynbioService

server_description = """
SynBio API
"""

app = FastAPI(
    title="SynBio",
    description=server_description,
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


def get_service():
    print('get_service')
    from minio import Minio
    from sqlalchemy import create_engine
    import yaml

    # Load the YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        client_mysql = create_engine(config['sql'])
        client_minio = Minio(config['minio']['host'],
                             secret_key=str(config['minio']['secret']),
                             access_key=str(config['minio']['access']), secure=False)
        service = SynbioService(client_minio, config['minio']['bucket'], client_mysql)
        return service


@app.get("/whoiam")
def get_whoiam():
    return JSONResponse(content="SynBio Server")


@app.post("/experiment/upload")
async def upload_experiment(
    file: UploadFile = File(...),
    experiment_id: str = Form(...),
    exp_index: int = Form(1),
    exp_type: str = Form('autoALE'),
    start_date: str = Form('2025-04-04'),
    lab_id: int = Form(1),
    contact_id: int = Form(1),
    description: str = Form(None),
    service: SynbioService = Depends(get_service)
):
    print('exp upload!')
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    content = await file.read()
    print(file.filename)
    try:
        service.add_plate_data(experiment_id, exp_index, exp_type,
                               file.filename, content,
                               start_date,
                               lab_id, contact_id,
                               description=description)
        df = pd.read_csv(io.BytesIO(content), sep=",")  # assuming tab-separated text file
        print(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{e}")

    experiment_id = experiment_id
    EXPERIMENTS[experiment_id] = {
        "description": description,
        "data": df
    }

    return {"experiment_id": experiment_id}


@app.get("/experiment/{exp_id}/strain/{strain_id}/query_od")
async def experiment_query_od(
    exp_id: str,
    strain_id: str,
    service: SynbioService = Depends(get_service)
):
    print(exp_id, strain_id)
    res = service.query_od(exp_id, strain_id)
    return JSONResponse(content=res.to_json())


@app.post("/lab")
async def add_lab(some_lab_parm: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/lab")
async def list_lab(
    service: SynbioService = Depends(get_service)
):
    return JSONResponse(content=service.lab_list().to_json())


@app.get("/lab/{lab_id}")
async def get_lab():
    return JSONResponse(content=list(["lab1", "lab2"]))


@app.post("/people")
async def add_people(some_lab_parm: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/people")
async def list_people(
    service: SynbioService = Depends(get_service)
):
    return JSONResponse(content=service.people_list().to_json())


@app.get("/people/{people_id}")
async def get_people():
    return JSONResponse(content=list(["lab1", "lab2"]))


@app.post("/protocol")
async def add_protocol(some_lab_parm: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/protocol")
async def list_protocol():
    return JSONResponse(content=list(["lab1", "lab2"]))


@app.get("/protocol/{protocol_id}")
async def get_protocol():
    return JSONResponse(content=list(["lab1", "lab2"]))


@app.post("/operation")
async def add_operation(some_lab_parm: str = Form(...)):
    # add new strain
    return JSONResponse(content=list(EXPERIMENTS))


@app.get("/operation")
async def list_operation():
    return JSONResponse(content=list(["lab1", "lab2"]))


@app.get("/operation/{operation_id}")
async def get_operation():
    return JSONResponse(content=list(["lab1", "lab2"]))


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
