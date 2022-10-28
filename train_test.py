import torch
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def eval_model_acc(model, data_loader, device):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.
    with torch.no_grad():  # Deactivate gradients for the following code
        for values, adjacency_matrices, data_labels in data_loader:
            values, adjacency_matrices, data_labels = values.to(device), adjacency_matrices.to(device), \
                                             data_labels.to(device)
            preds = model(values, adjacency_matrices)
            preds = preds.squeeze(dim=1)
            _, y_pred_tags = torch.max(preds, dim=1)
            y_pred_tags = y_pred_tags.cpu().detach().numpy()
            data_labels = data_labels.cpu().detach().numpy()
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            for y_pred_tag, data_label in zip(y_pred_tags, data_labels):
                if data_label == y_pred_tag:
                    true_preds += 1
                num_preds += 1

    acc = true_preds / num_preds
    return acc

def eval_model_auc(model, data_loader, device):
    model.eval() # Set model to eval mode
    true_preds, my_preds = [], []
    with torch.no_grad():  # Deactivate gradients for the following code
        for values, adjacency_matrices, data_labels in data_loader:
            values, adjacency_matrices, data_labels = values.to(device), adjacency_matrices.to(device), \
                                                      data_labels.to(device)
            preds = model(values, adjacency_matrices)
            preds = F.softmax(preds, dim=1)
            probabilities = preds[:, 1]
            true_preds += data_labels.tolist()
            my_preds += probabilities.tolist()

    metric_result = roc_auc_score(true_preds, my_preds)
    return metric_result


def calc_loss_validation(model, val_data_loader, loss_module, device):
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for values, adjacency_matrices, data_labels in val_data_loader:
            values, adjacency_matrices, data_labels = values.to(device), adjacency_matrices.to(device), \
                                             data_labels.to(device)
            preds = model(values, adjacency_matrices)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            # print(loss.item())
            val_running_loss += loss.item() * values.size(0)
    cur_val_loss = val_running_loss/len(val_data_loader.dataset)
    return cur_val_loss


def train_model(model, optimizer, train_data_loader, val_data_loader, loss_module, device, model_name, num_epochs=20):
    # Set model to train mode
    model.train()
    train_loss, val_loss, train_acc, val_acc, alpha_list = [], [], [], [], []
    # Training loop
    for epoch in range(num_epochs):
        train_running_loss = 0
        model.train()
        for values, adjacency_matrices, data_labels in train_data_loader:
            values, adjacency_matrices, data_labels = values.to(device), adjacency_matrices.to(device), \
                                             data_labels.to(device)
            preds = model(values, adjacency_matrices)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels)
            # print(loss.item())
            train_running_loss += loss.item() * values.size(0)
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero.
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            ## Step 5: Update the parameters
            optimizer.step()
        cur_train_loss = train_running_loss/len(train_data_loader.dataset)
        train_loss.append(cur_train_loss)
        cur_train_acc = eval_model_auc(model, train_data_loader, device)
        train_acc.append(cur_train_acc)
        cur_val_acc = eval_model_auc(model, val_data_loader, device)
        val_acc.append(cur_val_acc)
        cur_val_loss = calc_loss_validation(model, val_data_loader, loss_module, device)
        val_loss.append(cur_val_loss)
        if model_name == "just_values":
            print_msg = (f'[{epoch}/{num_epochs}] ' +
                         f'train_loss: {cur_train_loss:.9f} train_auc: {cur_train_acc:.9f} ' +
                         f'test_loss: {cur_val_loss:.6f} test_auc: {cur_val_acc:.6f} ')
        else:
            alpha_value = model.alpha.item()
            alpha_list.append(alpha_value)
            print_msg = (f'[{epoch}/{num_epochs}] ' +
                         f'train_loss: {cur_train_loss:.9f} train_auc: {cur_train_acc:.9f} ' +
                         f'test_loss: {cur_val_loss:.6f} test_auc: {cur_val_acc:.6f} ' +
                         f'Alpha value: {alpha_value} ')

        print(print_msg)

    return train_loss, val_loss, train_acc, val_acc, alpha_list


def plot_acc_loss(train_loss, val_loss, train_acc, val_acc, alpha_list, paramters_str):
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    x = range(len(train_loss))
    ax1.plot(x, train_loss, label="Train loss", color="red")
    ax1.plot(x, val_loss, label="Test loss", color="blue")
    ax1.title.set_text("Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(x)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.grid()

    ax1.legend()
    ax2.plot(x, train_acc, label="Train Accuracy", color="red")
    ax2.plot(x, val_acc, label="Test Accuracy", color="blue")
    ax2.title.set_text("Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(x)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.grid()

    ax2.legend()

    ax3.plot(x, alpha_list, label="Alpha value", color="black")
    ax3.title.set_text("Alpha")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Alpha value")
    ax3.set_xticks(x)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.grid()

    ax3.legend()
    fig.suptitle(f"{paramters_str}", fontsize=12)
    plt.tight_layout()
    dt_string2 = now.strftime("%d_%m_%Y_%H:%M:%S")
    plt.savefig(f"Toy_model_{paramters_str}.png")
    plt.show()
