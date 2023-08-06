import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f"{item:>6}")
    print(f"______\n{sum(params):>6}")


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    learning_rate=0.001,
    epochs=30,
    batch_size=32,
    val_batch_size=32,
    tolerance=25,
    lstm=False
):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, val_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()

    train_accuracies = []
    test_accuracies = []
    test_mean_losses = []
    train_mean_losses = []
    tolerance_count = 0

    for i in range(epochs):
        model.train()
        trn_corr = 0
        tst_corr = 0
        total_train_loss = 0
        total_test_loss = 0

        for b, (X_tr, y_tr) in enumerate(train_loader):
            b += 1
            if not lstm:
                y_pred = model(X_tr)
                loss = criterion(y_pred, y_tr)
                total_train_loss += loss.item()
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_tr).sum()
                trn_corr += batch_corr

            optimizer.zero_grad()
            if lstm:
                y_pred = model(X_tr)
                loss = criterion(y_pred, y_tr)
                total_train_loss += loss.item()
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_tr).sum()
                trn_corr += batch_corr
            loss.backward()
            optimizer.step()

        train_accuracy = (trn_corr.item() / len(train_dataset)) * 100
        train_mean_loss = total_train_loss / len(train_loader)
        train_accuracies.append(train_accuracy)
        train_mean_losses.append(train_mean_loss)

        model.eval()
        with torch.no_grad():
            for b, (X_ts, y_ts) in enumerate(test_loader):
                b += 1
                # Apply the model
                y_val = model(X_ts)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_ts).sum()

                val_loss = criterion(y_val, y_ts)
                total_test_loss += val_loss.item()

        test_accuracy = (tst_corr.item() / len(test_dataset)) * 100
        test_mean_loss = total_test_loss / len(test_loader)
        test_accuracies.append(test_accuracy)
        test_mean_losses.append(test_mean_loss)

        print(
            f"epoch:{i+1}\tTrain Loss:{train_mean_loss:12.2f}\tTrain Accuracy:{train_accuracy:12.2f}\tTest Loss:{test_mean_loss:12.2f}\tTest Accuracy:{test_accuracy:12.2f}\t Tolerance Count:{tolerance_count}"
        )

        last_few_train_losses = torch.round(
            torch.FloatTensor(train_mean_losses[-tolerance:]), decimals=3
        )
        train_loss_no_change = (
            torch.equal(
                last_few_train_losses[
                    last_few_train_losses[0] == last_few_train_losses
                ],
                last_few_train_losses,
            )
            and len(last_few_train_losses) == tolerance
        )
        last_few_test_losses = torch.round(
            torch.FloatTensor(test_mean_losses[-tolerance:]), decimals=3
        )
        test_loss_no_change = (
            torch.equal(
                last_few_test_losses[last_few_test_losses[0] == last_few_test_losses],
                last_few_test_losses,
            )
            and len(last_few_test_losses) == tolerance
        )

        if test_mean_loss >= train_mean_loss:
            tolerance_count += 1
            if tolerance_count >= tolerance:
                print("Early Stopping")
                print(f"\nDuration: {time.time() - start_time:.0f} seconds")
                return (
                    train_accuracies,
                    test_accuracies,
                    train_mean_losses,
                    test_mean_losses,
                )
        else:
            tolerance_count = 0

        if test_loss_no_change:
            print(f"No change in test loss for {tolerance} iterations")
            print(f"\nDuration: {time.time() - start_time:.0f} seconds")
            return (
                train_accuracies,
                test_accuracies,
                train_mean_losses,
                test_mean_losses,
            )
        elif train_loss_no_change:
            print(f"No change in train loss for {tolerance} iterations")
            print(f"\nDuration: {time.time() - start_time:.0f} seconds")
            return (
                train_accuracies,
                test_accuracies,
                train_mean_losses,
                test_mean_losses,
            )
        else:
            pass

    print(f"\nDuration: {time.time() - start_time:.0f} seconds")

    return (train_accuracies, test_accuracies, train_mean_losses, test_mean_losses)


def show_metrics(
    train_accuracies, test_accuracies, train_mean_losses, test_mean_losses
):
    metrics_dict = dict()
    metrics_dict["Training Accuracy"] = train_accuracies
    metrics_dict["Test Accuracy"] = test_accuracies
    metrics_dict["Train Loss"] = train_mean_losses
    metrics_dict["Test Loss"] = test_mean_losses
    df = pd.DataFrame(metrics_dict)
    plt.plot(df["Training Accuracy"], label="Training Accuracy")
    plt.plot(df["Test Accuracy"], label="Test Accuracy")
    plt.legend()
    plt.show()
    plt.plot(df["Train Loss"], label="Train Loss")
    plt.plot(df["Test Loss"], label="Test Loss")
    plt.legend()
    plt.show()


def eval_model(model, X_test, y_test,lstm=False):
    start = time.time()
    model.eval()
    test_data_set = TensorDataset(X_test, y_test)
    test_data_loader = DataLoader(test_data_set, len(test_data_set), shuffle=True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for b, (X_ts, y_ts) in enumerate(test_data_loader):
            y_val = model(X_ts)
            predicted = torch.max(y_val, 1)[1]
            loss = criterion(y_val, y_ts)
            total_loss += loss.item()
            total_correct += (predicted == y_ts).sum().item()
            print(classification_report(predicted,y_ts))
            print(confusion_matrix(predicted,y_ts))

    test_accuracy = (total_correct / len(test_data_set)) * 100
    test_mean_loss = total_loss / len(test_data_loader)
    print(f"\nDuration: {time.time() - start:.0f} seconds")
    return (test_accuracy, test_mean_loss)
