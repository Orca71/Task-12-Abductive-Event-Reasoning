def run_training(model, train_set, dev_set=None):
    """
    Call the underlying model's train function.
    """
    model.train(train_set, dev_set)