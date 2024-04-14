def prediccion_clasificacion(model, data):
    from pycaret.classification import predict_model
    return predict_model(model, data)

def prediccion_regresion(model, data):
    from pycaret.regression import predict_model
    return predict_model(model, data)