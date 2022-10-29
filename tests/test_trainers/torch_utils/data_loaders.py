import torch


def train_val_data_loaders(X, y, fold):
    X_train = torch.from_numpy(X[fold[0]]).float().to("cpu")
    y_train = torch.from_numpy(y[fold[0]]).float().to("cpu")
    X_test = torch.from_numpy(X[fold[1]]).float().to("cpu")
    y_test = torch.from_numpy(y[fold[1]]).float().to("cpu")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=30_000, shuffle=False
    )

    val_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=30_000, shuffle=False
    )

    return (train_loader, val_loader)
