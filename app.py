from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import json
from main import get_prediction

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/predict')
async def get_hotels(loss_of_capacity: int, managnese_cathode_voltage_conduction: int, temperature_during_high_performance: int, cobalt_anode_voltage_conduction: int, internal_resistance: int, internal_humidity: int):
    return json.loads(get_prediction(loss_of_capacity, managnese_cathode_voltage_conduction, temperature_during_high_performance, cobalt_anode_voltage_conduction, internal_resistance, internal_humidity).to_json(orient='records'))[0]
