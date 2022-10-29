import pytorch_lightning as pl


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.validation_metric = []
        self.n_epochs = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_metric.append(trainer.logged_metrics["validation_metric"])
        self.n_epochs += 1

    def get_metric(self):
        return self.validation_metric

    def get_n_epochs(self):
        return self.n_epochs
