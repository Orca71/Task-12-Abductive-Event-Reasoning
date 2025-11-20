from .base_model import BaseModel

class TrainableModel(BaseModel):
    """
    Models that support fine-tuning should inherit from this.
    """

    def train(self, train_set, dev_set=None):
        """
        Train the model on given dataset(s).
        """
        raise NotImplementedError