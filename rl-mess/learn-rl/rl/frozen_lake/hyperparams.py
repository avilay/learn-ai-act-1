from configparser import ConfigParser


class Hyperparams:
    def __init__(self, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.lr = kwargs["lr"]
        self.num_train_steps = kwargs["num_train_steps"]
        self.num_val_steps = kwargs["num_val_steps"]
        self.activation = kwargs["activation"]

    @staticmethod
    def csv_header():
        return "batch,epochs,lr,n_train,n_val,act"

    def csv_repr(self):
        return f"{self.batch_size},{self.epochs},{self.lr},{self.num_train_steps},{self.num_val_steps},{self.activation}"

    def __repr__(self):
        return f"""
Hyperparams -
    batch_size={self.batch_size}
    epochs={self.epochs}
    lr={self.lr}
    num_train_steps={self.num_train_steps}
    num_val_steps={self.num_val_steps}
    activation={self.activation}
"""

    @classmethod
    def load(cls, conffile):
        conf = ConfigParser()
        conf.read(conffile)
        return cls(
            batch_size=conf["HYPERPARAMS"].getint("batch_size"),
            epochs=conf["HYPERPARAMS"].getint("epochs"),
            lr=conf["HYPERPARAMS"].getfloat("lr"),
            num_train_steps=conf["HYPERPARAMS"].getint("num_train_steps"),
            num_val_steps=conf["HYPERPARAMS"].getint("num_val_steps"),
            activation=conf["HYPERPARAMS"].get("activation"),
        )
