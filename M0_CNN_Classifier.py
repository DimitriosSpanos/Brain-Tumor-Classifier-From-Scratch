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

# Hyper-Parameters
epochs = 150
img_size = 28
lr = 0.01
n1 = 200
n2 = 100
n3 = 50
decay = 1 / 1.3

def main():
    # Creation of trn,tst datasets
    trn_x,trn_y, tst_x,tst_y = getData()
    print("Training:", len(trn_y), "instances.")
    print("Testing:", len(tst_y), "instances.")

    kernel_size = 3
    side_after_conv = img_size-kernel_size+1
    # Creation of the Model
    model = [
        my_Conv2d(input_shape=(1, img_size, img_size), kernel_size=kernel_size,  output_channels=3, initialization='He'),
        my_Tanh(),

        my_Flatten(input_shape=(3, side_after_conv, side_after_conv), output_shape=(3 * side_after_conv * side_after_conv, 1)),

        my_Linear(input_size=3 * side_after_conv * side_after_conv, output_size=n1, initialization='He'),
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
    print((train(model,trn_x,trn_y, tst_x,tst_y,lr)))

    print(f"\nTraining took {((time.time() - start_time) / 60):.4f} minutes.\n")



def train(model, trn_x, trn_y, tst_x, tst_y, learning_rate):
    """
    :param model: the model to be trained
    :param trn_x: the training input data
    :param trn_y: the training output data
    :param tst_x: the test input data
    :param tst_y: the test output data
    :param learning_rate: the learning rate to be used for training
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

    drawLearningCurves(losses, test_losses, train_accuracies, test_accuracies)
    drawConfusionMatrix(targets,predictions,'From_Scratch_Confusion.png')

    return tst_accuracy


def drawLearningCurves(train_losses, test_losses, train_accuracies, test_accuracies):
    """
    :param train_losses: a history of training losses as a list
    :param test_losses: a history of test losses as a list
    :param train_accuracies: a history of training accuracies as a list
    :param test_accuracies: a history of testing accuracies as a list
    :return:
    """
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