"""
MetricsVerbosity.TRAIN_METRICS will calculate both train and val metrics for each epoch
                .EPOCH_VAL_METRICS will calculate val metrics for each epoch
                .FINAL_VAL_METRICS will calculate val metrics for only the final epoch

trainer.train will always save the model at the end of the training run
trainer.auto_tune will save only the final best model and run name for that will be best_model
"""


@abstractclass
class Metrc:
    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def calc(self, outputs, targets):
        pass


def learn(hparams, trainset, valset):
    return TrainerArgs(
        run_name=Haikunator().haikunate(),
        model=create_model(),
        optim=t.optim.SGD(model.parameters(), lr=hparams.learning_rate),
        loss_fn=t.nn.CrossEntropyLoss(),
        trainloader=t.utils.data.DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True),
        valloader=t.utils.data.DataLoader(valset, batch_size=5000),
    )


def main_simple():
    trainset = ...
    valset = ...
    testset = ...
    metrics_config = MetricsConfig(
        verbosity=MetricsVerbosity.TRAIN_METRICS, log_metrics=True, metrics=[Accuracy(), F1Score()]
    )
    trainer = Trainer(EXPERIMENT_NAME, trainset, valset, metrics_config, clip_grads=0.9)

    hparams = Hyperparams()
    model = trainer.train(hparams, learn)
    final_val_loss = trainer.val_losses[-1]
    final_val_acc = trainer.metric(Accuracy.name, Trainer.VALSET)[-1]

    test_metrics = trainer.evaluate(model, testset)
    test_metrics[Accuracy.name]


def main_autotune():
    trainset = ...
    valset = ...
    testset = ...
    metrics_config = MetricsConfig(
        verbosity=MetricsConfig.FINAL_VAL_METRICS, log_metrics=False, metrics=[Accuracy()]
    )
    trainer = Trainer(EXPERIMENT_NAME, trainset, valset, metrics_config)
    hparams_spec = [
        {"name": "batch_size", "type": "choice", "value_type": "int", "values": [16, 32, 64]},
        {"name": "epochs", "type": "range", "value_type": "int", "bounds": [7, 13]},
        {"name": "learning_rate", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
    ]
    model = trainer.auto_tune(hparams_spec, learn, objective_metric=Accuracy.name)
    trainer.best_params
