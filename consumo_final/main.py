import pandas as pd
from cargar_modelos import *
from tratamiento_datos import *
from predicciones import *
import re
if __name__ == "__main__":
    from pathlib import Path

    # Ruta de la carpeta a recorrer
    clasificacion_path = Path("../models/clasificacion")
    regresion_path = Path("../models/regresion")
    # Recorrer todos los archivos en la carpeta y sus subcarpetas
    path = "../data/transformed/cargas_binarias_feature_eng.csv"
    df_original = pd.read_csv(path)
    df_class = procesar_datos_clasificacion(path)[:60]
    try:
        df_class.drop(['Refrigerator',  
            'Clothes washer',
            'Clothes Iron',
            'Computer',
            'Oven',
            'Play',
            'TV',
            'Sound system'], axis=1, inplace=True)
    except:
        pass
    df_reg = df_class.copy()

    for ruta_archivo in clasificacion_path.glob('**/*'):
        if ruta_archivo.is_file():
            print()
            model = cargar_modelo_clasificacion(str(ruta_archivo)[:-4])
            df_pred = prediccion_clasificacion(model, df_class)
            df_reg[re.search(r"tuned_(.*?)\.pkl", str(ruta_archivo)).group(1).replace('_', ' ')] = df_pred["prediction_label"]
            
    print(df_reg)
    df_desagregado = df_original.copy()
    for ruta_archivo in regresion_path.glob('**/*'):
        if ruta_archivo.is_file():
            y_pred = prediccion_regresion(cargar_modelo_regresion(str(ruta_archivo)[:-4]), df_reg)
            df_desagregado[re.search(r"modelo (.*?)\.pkl", str(ruta_archivo)).group(1)] = y_pred["prediction_label"]

    df_desagregado[["Fecha", "Medidor [W]", 'Refrigerator',  
            'Clothes washer',
            'Clothes Iron',
            'Computer',
            'Oven',
            'Play',
            'TV',
            'Sound system']].to_csv("predictions.csv")
    df_reg[["Fecha", "Medidor [W]", 'Refrigerator',  
            'Clothes washer',
            'Clothes Iron',
            'Computer',
            'Oven',
            'Play',
            'TV',
            'Sound system']].to_csv("predictions.csv")
    print(df_desagregado)