import shap

explainer = None

def init_explainer(model, X_sample):
    global explainer
    explainer = shap.Explainer(model, X_sample)

def get_shap_values(input_data):
    global explainer
    return explainer(input_data)
