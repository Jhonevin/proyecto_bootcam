from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os


# Cargar el modelo y el escalador entrenados
modelo = joblib.load("modelo_knn.pkl")
scaler = joblib.load("scaler.pkl")

# Inicializar la app
app = FastAPI(title="API de Predicción de Inundaciones", version="1.0")

# Esquema de datos de entrada según las variables usadas
class DatosEntrada(BaseModel):
    precipitacion_mm: float
    temperatura_C: float
    duracion_lluvia: float
    indice_humedad: float
    estado_climatico: float

# Ruta principal (opcional)
@app.get("/")
def inicio():
    return {
        "mensaje": "Bienvenido a la API de Predicción de Inundaciones",
        "usar": "/predict para enviar datos",
        "ver historial": "/historial para ver registros"
    }

# Ruta de predicción (POST)
@app.post("/predict")
def hacer_prediccion(datos: DatosEntrada):
    # Convertir los datos a arreglo NumPy
    datos_np = np.array([[ 
        datos.precipitacion_mm,
        datos.temperatura_C,
        datos.duracion_lluvia,
        datos.indice_humedad,
        datos.estado_climatico
    ]])

    # Escalar los datos
    datos_escalados = scaler.transform(datos_np)

    # Realizar predicción
    prediccion = modelo.predict(datos_escalados)[0]

    # Preparar fila para guardar
    fila = {
        "fecha_hora": datetime.now().isoformat(),
        "precipitacion_mm": datos.precipitacion_mm,
        "temperatura_C": datos.temperatura_C,
        "duracion_lluvia": datos.duracion_lluvia,
        "indice_humedad": datos.indice_humedad,
        "estado_climatico": datos.estado_climatico,
        "prediccion": int(prediccion)
    }

    # Guardar la predicción en CSV
    archivo = "registro_predicciones.csv"
    if os.path.exists(archivo):
        df_existente = pd.read_csv(archivo)
        df_nuevo = pd.concat([df_existente, pd.DataFrame([fila])], ignore_index=True)
    else:
        df_nuevo = pd.DataFrame([fila])
    
    df_nuevo.to_csv(archivo, index=False)

    return {"prediccion": int(prediccion)}

# Ruta para ver el historial de predicciones
@app.get("/historial")
def ver_historial():
    archivo = "registro_predicciones.csv"
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        return df.tail(10).to_dict(orient="records")  # Muestra las últimas 10
    else:
        return {"mensaje": "Aún no hay registros de predicciones."}
