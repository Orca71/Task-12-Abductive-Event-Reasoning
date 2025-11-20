def run_inference(model, dataset):
    """
    Apply a model to a dataset (list of items).
    Returns list of predictions.
    """
    preds = []
    for item in dataset:
        pred = model.predict(item)
        preds.append(pred)
    return preds