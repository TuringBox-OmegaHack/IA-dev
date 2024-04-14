def cargar_modelo_clasificacion(path):
    from pycaret.classification import load_model
    return load_model(path)

def cargar_modelo_regresion(path):
    from pycaret.regression import load_model
    return load_model(path)