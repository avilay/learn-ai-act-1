import pytorch_lightning as pl
import torch as t
from torchmetrics import Accuracy


class LitImdb(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = t.nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy("binary")
        self.val_acc = Accuracy("binary")

        self.val_acc_value = 0.0

        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        return t.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.train_acc(outputs, targets)

        self.log("train_step_loss", loss)
        self.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        self.log("acc", {"train_acc": self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.val_acc.update(outputs, targets)

        # By default this is accumulated at every step and logged at epoch end
        self.log("loss", {"val_loss": loss})
        self.log("acc", {"val_acc": self.val_acc})

        # logging this separately because it is needed during ax tuning
        self.log("val_acc", self.val_acc)
        return {}
