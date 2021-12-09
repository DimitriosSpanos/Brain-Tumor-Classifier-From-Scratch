"""
Neural Networks - Deep Learning
MRI Brain Tumor Classifier ('pituitary' | 'no tumor' | 'meningioma' | 'glioma')
Author: Dimitrios Spanos Email: dimitrioss@ece.auth.gr
"""
from M0 import my_Conv2d, my_Linear, my_Flatten, my_Sigmoid, my_Tanh, my_MSE, my_d_MSE
from M0 import my_Softmax, my_d_cross_entropy, my_cross_entropy
from M0_auxiliary import to_categorical
import numpy as np
from torchvision import datasets, transforms # used ONLY for acquisition of the dataset
from torch.utils.data import DataLoader # used ONLY for acquisition of the dataset
import time, os, warnings
import seaborn as sn # for heatmaps
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

root = "cleaned/"
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = 28

# Hyper-Parameters
epochs = 150
# Grid Search 2x2x1x2x2
lr_values = [0.1,0.01]
n1_values = [200,400]
n2_values = [100]
n3_values = [25,50]
decay_values = [1/1.3, 1/1.5]

def main():
    # Creation of trn,tst datasets
    trn_x,trn_y, tst_x,tst_y = getData()
    print("Training:", len(trn_y), "instances.")
    print("Testing:", len(tst_y), "instances.")

    # GRID SEARCH
    accuracies,grid = [],[]
    grid_start = time.time()
    for n1 in n1_values:
        for lr in lr_values:
            for n2 in n2_values:
                for n3 in n3_values:
                    for decay in decay_values:
                        # Creation of the Model
                        model = [
                            my_Flatten(input_shape=(1, img_size, img_size), output_shape=(1 * img_size * img_size, 1)),

                            my_Linear(input_size=1 * img_size * img_size, output_size=n1, initialization='He'),
                            my_Tanh(),

                            my_Linear(input_size=n1, output_size=n2, initialization='He'),
                            my_Tanh(),

                            my_Linear(input_size=n2, output_size=n3, initialization='He'),
                            my_Tanh(),

                            my_Linear(input_size=n3, output_size=4, initialization='He'),
                            my_Softmax()
                        ]
                        # Training of the Model
                        start_time = time.time()
                        accuracies.append(grid_search(model,trn_x,trn_y, tst_x,tst_y,lr,decay))
                        grid.append((lr,n1,n2,n3,decay))
                        print(f"\nTraining one model takes {((time.time() - start_time) / 60):.4f} minutes.\n")
    grid_end = time.time()
    print(f"\nGrid Search took {((grid_end - grid_start) / 3600):.2f} hours.")
    max_accuracy = max(accuracies)
    for i, acc in enumerate(accuracies):
        if acc == max_accuracy:
            print('--- Grid Search Results ---')
            print(f'lr={grid[i][0]}, n1={grid[i][1]}, n2={grid[i][2]}, n3={grid[i][3]}, decay={grid[i][4]}')
    train(accuracies,grid)


def grid_search(model, trn_x, trn_y, tst_x, tst_y,learning_rate,decay):
    """
    :param model: the model to be trained
    :param trn_x: the training input data
    :param trn_y: the training output data
    :param tst_x: the test input data
    :param tst_y: the test output data
    :param learning_rate: the learning rate to be used for training
    :param decay: the factor the learning rate decreases by every 10 epochs
    """
    losses,test_losses,train_accuracies,test_accuracies,targets,predictions = [],[],[],[],[],[]

    for epoch in range(epochs):
        """
        --- Learning Rate scheduler ---
        Every 10 epochs the learning rate is 
        decreased by a factor of "decay".
        """
        if epoch % 10 == 0:
            learning_rate *= decay
        epoch_start = time.time()
        loss,tst_loss,num_correct,num_samples,tst_num_correct,tst_num_samples = 0,0,0,0,0,0


        # TRAINING
        for i, x in enumerate(trn_x):
            y = np.reshape(trn_y[i], (-1,1))

            # forward
            y_pred = x
            for layer in model:
                y_pred = layer.forward(y_pred)

            # error
            loss += my_cross_entropy(y, y_pred)

            # Tally the number of correct predictions
            num_samples += 1
            if np.argmax(y_pred) == np.argmax(y):
                num_correct += 1

            # backward
            grad = my_d_cross_entropy(y, y_pred)
            for layer in reversed(model):
                grad = layer.backward(grad, learning_rate)

        # Testing
        for i, x in enumerate(tst_x):

            y = np.reshape(tst_y[i], (-1,1))
            y_pred = x
            for layer in model:
                y_pred = layer.forward(y_pred)

            tst_loss += my_cross_entropy(y, y_pred)

            # Tally the number of correct predictions
            tst_num_samples += 1
            if np.argmax(y_pred) == np.argmax(y):
                tst_num_correct += 1

            # only used for drawing the confusion matrix
            if epoch == epochs - 1:
                targets.append(np.argmax(y))
                predictions.append( np.argmax(y_pred))

        loss /= len(trn_y)
        tst_loss /= len(tst_y)
        trn_accuracy = float(num_correct) / float(num_samples) * 100
        tst_accuracy = float(tst_num_correct) / float(tst_num_samples) * 100
        epoch_end = time.time()

        # keep track of the metrics of each epoch ( for learning curves )
        losses.append(loss)
        test_losses.append(tst_loss)
        train_accuracies.append(trn_accuracy)
        test_accuracies.append(tst_accuracy)

        print(f"Epoch {(epoch + 1):02}/{epochs} => {int(epoch_end-epoch_start)}s - loss: {loss:.5f} - accuracy: {trn_accuracy:.2f}% - test_loss: {tst_loss:.5f} - test_accuracy: {tst_accuracy:.2f}%")

        """
                --- Early Stopping ---
        If the model has trained for 10 epochs
        and it hasn't passed a certain test
        accuracy threshold, the training is terminated
        """
        if epoch >= 9 and tst_accuracy < 50:
            return tst_accuracy
    drawLearningCurves(losses, test_losses, train_accuracies, test_accuracies)
    drawConfusionMatrix(targets,predictions,'From_Scratch_Confusion.png')

    return tst_accuracy

def train(accuracies,grid):
    """
    :param accuracies: all the accuracies of the grid in a list
    :param grid: the grid containing the hyper-parameters
    """
    max_accuracy = max(accuracies)
    for i, acc in enumerate(accuracies):
        if acc == max_accuracy:

            print('--- Final Training ---')
            model = [
                my_Flatten(input_shape=(1, img_size, img_size), output_shape=(1 * img_size * img_size, 1)),

                my_Linear(input_size=1 * img_size * img_size, output_size=grid[i][1], initialization='He'),
                my_Tanh(),

                my_Linear(input_size=grid[i][1], output_size=grid[i][2], initialization='He'),
                my_Tanh(),

                my_Linear(input_size=grid[i][2], output_size=grid[i][3], initialization='He'),
                my_Tanh(),

                my_Linear(input_size=grid[i][3], output_size=4, initialization='He'),
                my_Softmax()
            ]
            print(f'Achieved Test Accuracy = {grid_search(model, trn_x, trn_y, tst_x, tst_y, grid[i][0], grid[i][4])}%')

def drawLearningCurves(train_losses, test_losses, train_accuracies, test_accuracies):

    # loss
    plt.plot(train_losses, color='g')
    plt.plot(test_losses, color='b')
    plt.title('Loss - Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train_loss', 'test_loss'], loc='upper right')
    plt.savefig('From_Scratch_Loss.png', dpi=400, bbox_inches='tight')
    plt.close()

    # accuracy
    plt.plot(train_accuracies, color='g')
    plt.plot(test_accuracies, color='b')
    plt.title('Accuracy - Learning curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train_accuracy %', 'test_accuracy %'], loc='lower right')
    plt.savefig('From_Scratch_Accuracy.png', dpi=400, bbox_inches='tight')
    plt.close()


def drawConfusionMatrix(tst_y, y_pred, name):

    # Draw the Confusion Matrix
    cf_matrix = confusion_matrix(tst_y, y_pred)
    plt.title('Brain Tumor Confusion Matrix')
    sn.set(font_scale=0.6)
    sn.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.savefig(name, dpi=550, bbox_inches='tight')
    plt.close()


def getData():
    """
    :return: The datasets in numpy form
    """
    tranforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(os.path.join(root, "Training"), transform=tranforms)
    test_data = datasets.ImageFolder(os.path.join(root, "Testing"), transform=tranforms)

    trn,tst = DataLoader(train_data, num_workers=4),DataLoader(test_data, num_workers=4)

    np_trn_x,np_trn_y, np_tst_x,np_tst_y = [], [], [],[]

    for i in range(len(trn.dataset)):
        np_trn_x.append(trn.dataset[i][0].detach().numpy())
        np_trn_y.append(trn.dataset[i][1])

    for i in range(len(tst.dataset)):
        np_tst_x.append(tst.dataset[i][0].detach().numpy())
        np_tst_y.append(tst.dataset[i][1])

    np_trn,np_tst = list(zip(np_trn_x,np_trn_y)),list(zip(np_tst_x, np_tst_y))
    np.random.shuffle(np_trn)
    np.random.shuffle(np_tst)

    np_trn_x, np_trn_y = zip(*np_trn)
    np_tst_x, np_tst_y = zip(*np_tst)

    np_trn_y,np_tst_y  = to_categorical(np_trn_y),to_categorical(np_tst_y)

    return np_trn_x,np_trn_y, np_tst_x,np_tst_y


if __name__ == '__main__':
    main()