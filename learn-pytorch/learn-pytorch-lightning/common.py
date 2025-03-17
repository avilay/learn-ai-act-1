from haikunator import Haikunator
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import wandb
import warnings
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning


warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=".*data_loading.*")


def train(
    project,
    run_name,
    model,
    data,
    hparams,
    callbacks=None,
    use_csv_logger=False,
    **trainer_flags,
):
    run_name = run_name if run_name else Haikunator().haikunate()

    print(f"Starting run {project}/{run_name}")
    runroot = Path.home() / "mlruns"

    if use_csv_logger:
        logger = CSVLogger(save_dir=runroot, name=project)
    else:
        logger = WandbLogger(
            project=project,
            name=run_name,
            save_dir=runroot,
            log_model="all",
            id=run_name,
        )
        logger.watch(model, log="all")

    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    callbacks = [checkpoint] + callbacks if callbacks else [checkpoint]
    print(f"APTG DEBUG: {callbacks}")
    trainer = Trainer(
        default_root_dir=runroot,
        max_epochs=hparams.n_epochs,
        logger=logger,
        callbacks=callbacks,
        **trainer_flags,
    )

    trainer.fit(model, data)
    return trainer


def test(project, run_name, model_cls, data_cls, upload_logs=True, **trainer_flags):
    run = wandb.init(project=project)

    model_artifact = f"avilay/{project}/model-{run_name}:best"
    artifact = run.use_artifact(model_artifact, type="model")
    artifact_dir = artifact.download()

    model_checkpoint = Path(artifact_dir) / "model.ckpt"
    if not model_checkpoint.exists():
        raise RuntimeError(f"Model checkpoint at {model_checkpoint} does not exist!")

    model = model_cls.load_from_checkpoint(model_checkpoint)
    data = data_cls()
    if upload_logs:
        logger = WandbLogger()
        Trainer(logger=logger, **trainer_flags).test(model, data)
    else:
        Trainer(**trainer_flags).test(model, data)
