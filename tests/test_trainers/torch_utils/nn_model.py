import pytorch_lightning as pl
import torch


class nn_model(pl.LightningModule):
    def __init__(
        self,
        input_size,
        embedding_size,
        objective,
        validation_metric,
        lr,
        weight_decay,
        optimizer,
    ):
        super(nn_model, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.linear1 = torch.nn.Linear(input_size, embedding_size)
        self.linear2 = torch.nn.Linear(embedding_size, 1)
        if objective == "MSE":
            self.objective = torch.nn.MSELoss()
        self.validation_metric = validation_metric

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        y_pred = self.forward(x)
        y_pred = y_pred.view(-1)
        train_loss = self.objective(y_pred, y)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        y_pred = self.forward(x)
        val_loss = self.validation_metric(y_pred, y)
        self.log("validation_metric", val_loss)
        return val_loss

    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "Adadelta":
            return torch.optim.Adadelta(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "Adagrad":
            return torch.optim.Adagrad(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "Adam":
            return torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "Adamax":
            return torch.optim.Adamax(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "ASGD":
            return torch.optim.ASGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "LBFGS":
            return torch.optim.LBFGS(self.parameters(), lr=self.lr)
        elif self.optimizer == "RMSprop":
            return torch.optim.RMSprop(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "Rprop":
            return torch.optim.Rprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
