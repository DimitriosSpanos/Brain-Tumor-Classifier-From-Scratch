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
import numpy as np
import seaborn as sn # for heatmaps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "cleaned/"
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
# Hyper-Parameters acquired from Grid Search
learning_rate = 5e-4
p = 0.5
neurons = 2048
batch_size = 128
epochs = 30

def main():

    # Creation of Data Loaders
    train_loader, test_loader = getData()

    # Creation of the final model
    model = M1(hidden_neurons=neurons, dropout_probability = p)

    # Train the final model
    trained_model = train(model, train_loader)

    # Test the trained model
    test(trained_model,test_loader)


def train(model, train_loader):
    """
    :param model: The untrained model
    :param train_loader: The train loader containing the train_X, train_y
    :return: The trained model
    """
    if device == torch.device("cuda"):
        model.cuda()

    model.train()
    criterion = nn.CrossEntropyLoss()
    if device == torch.device("cuda"):
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.2)
    start_time = time.time()

    loss,val_loss = 0,0
    # Training
    for epoch in range(epochs):

        num_correct, num_samples,val_num_correct,val_num_samples = 0,0,0,0
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

        # Results
        scheduler.step()
        trn_accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Epoch {(epoch+1):02}/{epochs} => loss: {loss.item():.5f} - accuracy: {trn_accuracy:.2f}%")

    total = time.time() - start_time
    print(f"\nFinal training took {(total / 60):.4f} minutes.\n")
    return model


def test(trained_model, test_loader):
    """
    :param trained_model: The trained model
    :param test_data: The test dataset
    """
    predictions,targets = [],[]
    if device == torch.device("cuda"):
        trained_model.cuda()
    trained_model.eval()
    num_correct = 0
    num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for x_test, y_test in test_loader:

            x_test, y_test = x_test.to(device=device), y_test.to(device=device)
            y_pred = trained_model(x_test)


            predicted =  torch.max(y_pred.data, 1)[1]
            for i in predicted:
                predictions.append(predicted[i].cpu())
                targets.append(y_test[i].cpu())
            num_correct += (predicted == y_test).sum()
            num_samples += len(predicted)

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Testing accuracy: {accuracy:.2f}%")
    drawConfusionMatrix(targets, predictions, 'CNN_Confusion_matrix.png')
    total = time.time() - start_time
    print(f"Testing took {total:.5f}secs.\n{(total/float(num_samples)*1000):.5f}ms per image on average.")


def drawConfusionMatrix(tst_y, y_pred, name):
    """
    :param tst_y: The target labels as a list
    :param y_pred: The prediction labels as a list
    :param name: The name of the confusion matrix png file to be saved
    """
    # Draw the Confusion Matrix
    cf_matrix = confusion_matrix(tst_y, y_pred)
    plt.title('Brain Tumor Confusion Matrix')
    sn.set(font_scale=0.6)
    sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.savefig(name, dpi=550, bbox_inches='tight')
    plt.close()

def getData():
    """
    :return: The train, test loaders
    """
    tranforms = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(root, "Training"), transform=tranforms)
    test_data = datasets.ImageFolder(os.path.join(root, "Testing"), transform=tranforms)

    trn = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    tst = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    return trn, tst

if __name__ == '__main__':
    main()


