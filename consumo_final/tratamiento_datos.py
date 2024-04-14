import pandas as pd

def procesar_datos_clasificacion(data):

    df_original = pd.read_csv(data)
    df_original.drop("Unnamed: 0", axis=1, inplace=True)
    df_original["Fecha"] = pd.to_datetime(df_original["Fecha"])
    df_original["Month"] = df_original["Fecha"].dt.month.astype("object")
    df_original["Day"] = df_original["Fecha"].dt.day.astype("object")
    df_original["Hour"] = df_original["Fecha"].dt.hour.astype("object")
    df_original["Minutes"] = df_original["Fecha"].dt.minute.astype("object")
    df_original["WeekDay"] = df_original["Fecha"].dt.day_name()
    df_original["WeekDayNumber"] = df_original["Fecha"].dt.day_of_week.astype("object")
    df_original["Ptc_delta"] = df_original["Medidor [W]"].pct_change() * 100
    df_original["Movil"] = df_original["Medidor [W]"].rolling(window=3).mean()
    df_original["Hora_punta"] = pd.cut(df_original["Hour"], [-1, 9, 12, 18, 21, 23], labels=["No pico", "Pico", "No pico", "Pico", "No pico"], ordered=False)
    df_original["delta"] = df_original["Medidor [W]"].diff()
    return df_original
    
def concatenar_pd(dfs):
    df_concatenado = pd.concat(dfs, axis=1)
    
def procesar_datos_regresion(data):
    from pycaret.classification import load_model as load_classification
    from predicciones import prediccion_clasificacion
    from pathlib import Path

    # Ruta de la carpeta a recorrer
    clasificacion_path = Path("../models/clasificacion")
    
    for ruta_archivo in clasificacion_path.glob('**/*'):
        if ruta_archivo.is_file():
            y_pred = prediccion_clasificacion(load_classification(str(ruta_archivo)[:-4]))