"""
Neural Networks - Deep Learning
MRI Brain Tumor Classifier ('pituitary' | 'no tumor' | 'meningioma' | 'glioma')
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""

from M1 import M1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import time, os, warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "cleaned/"

# GRID SEARCH
# Grid with a size of 2x2x2x2 = 16
lr_values = [5e-3,5e-4]
p_values = [0.3,0.5]
neuron_values = [1024,2048]
bs_values = [32,128]
epochs = 30

def main():
    # Creation of datasets
    train_dataset, test_dataset = getData()

    grid, accuracies = [],[]
    start_time = time.time()
    # Grid Search
    for lr in lr_values:
        for p in p_values:
            for neurons in neuron_values:
                for batch_size in bs_values:

                    model,val_accuracy = cross_validate(train_dataset, lr, batch_size, neurons, p)

                    grid.append((lr,p,neurons,batch_size))
                    accuracies.append(val_accuracy)

    end_time = time.time()
    total = end_time - start_time
    print(f"Grid search(with k-Fold Cross Validation) took {(total / 3600):.2f} hours.")

    # Printing the best combination of the grid
    for i, (lr,p,neurons,batch_size) in enumerate(grid):
        if max(accuracies) == accuracies[i]:
            print(f'Maximum Validation Accuracy= {max(accuracies):.2f}%')
            print(f'Achieved with: learning rate= {lr}, dropout probability= {p}, hidden neurons={neurons}, batch size={batch_size}')



def cross_validate(train_dataset, lr, batch_size, neurons, p):
    """
    :param train_dataset: The train dataset
    :param lr: The learning rate used for training
    :param batch_size: The number of instances looked at simultaneously
    :param neurons: The units of the hidden layer
    :param p: The dropout rate of the two Dropout Layers
    :return: The trained model, validation accuracy
    """

    criterion = nn.CrossEntropyLoss()
    if device == torch.device("cuda"):
        criterion = criterion.cuda()

    start_time = time.time()
    # KFold Cross Validation - 80%-20% split
    kFold = KFold(n_splits=5, shuffle=True)
    kFold.get_n_splits(train_dataset)
    crossval_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kFold.split(train_dataset)):
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers = 2)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers = 2)

        # Creation of the model
        model = M1(hidden_neurons=neurons, dropout_probability = p)
        if device == torch.device("cuda"):
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.2)

        model.train()
        loss,val_accuracy = 0,0

        # Training
        for epoch in range(epochs):

            num_correct, num_samples = 0,0
            for (X_train, y_train) in train_loader:

                X_train, y_train = X_train.to(device), y_train.to(device)
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

                # Tally the number of correct predictions
                predicted = torch.max(y_pred.data, 1)[1]
                num_correct += (predicted == y_train).sum()
                num_samples += len(predicted)

                # Update Parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            val_loss, val_accuracy = validate_model(model, val_loader, criterion)

            # Results
            trn_accuracy = float(num_correct) / float(num_samples) * 100
            print(f"Epoch {(epoch+1):02}/{epochs} => loss: {loss.item():.5f} - accuracy: {trn_accuracy:.2f}% - val_loss: {val_loss.item():.5f} - val_accuracy: {val_accuracy:.2f}%")

        # The last validaction accuracy gets appended to the list containing the 5 different accuracies
        crossval_accuracies.append(val_accuracy)

    # We keep the average of the k accuracies as the model's final validation acccuracy
    mean_accuracy = sum(crossval_accuracies) / len(crossval_accuracies)

    total = time.time() - start_time
    print(f"\nk-Fold Cross Validation took {(total / 60):.4f} minutes.\n")
    return model, mean_accuracy


def validate_model(model, val_loader, criterion):
    """
    :param model: The model to be validated
    :param val_loader: The DataLoader containing the validation dataset
    :param criterion: The chosen loss function (CrossEntropy)
    :return: The validation loss, validation accuracy
    """
    model.eval()
    val_num_correct, val_num_samples, val_loss = 0, 0, 0

    with torch.no_grad():
        for (X_val, y_val) in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            y_pred = model(X_val)
            val_loss = criterion(y_pred, y_val)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            val_num_correct += (predicted == y_val).sum()
            val_num_samples += len(predicted)

    val_accuracy = float(val_num_correct) / float(val_num_samples) * 100
    model.train()
    return val_loss, val_accuracy


def getData():
    """
    :return: The train, test datasets
    """
    tranforms = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(root, "Training"), transform=tranforms)
    test_data = datasets.ImageFolder(os.path.join(root, "Testing"), transform=tranforms)

    return train_data, test_data

if __name__ == '__main__':
    main()


