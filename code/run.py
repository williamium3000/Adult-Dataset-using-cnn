import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import copy
import os
import time
import json
import logging
import cnn
import dataLoader
logging.basicConfig(filename="test_run_rec.log", level=logging.INFO)
def check_accuracy(device, loader, model, phase):
    logging.info('Checking accuracy on %s set: ' % phase)
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        if phase == "train":
            for x, y in loader:
                x = x.to(device=device).float()  # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
        elif phase == "val":
            for x, y in loader:
                x = x.to(device=device).float()   # move to device, e.g. GPU
                y = y.to(device=device)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        logging.info('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc



def train(model, optimizer, dataloaders, device, epochs):
    """
    Inputs:
        - model: A PyTorch model to train.
        - optimizer: An Optimizer object we will use to train the model.
        - dataloaders: dataLoaders of the data.
        - device: gpu device(or cpu) on which the training process is on.
        - epochs: the total number of epochs to train.
    Returns: 
        - model: Model with best test acc
        - best_acc: the val accuracy of the best model
        - rec: the information of the training process
            - loss of each epochs
            - train acc of each epochs
            - val acc of each epochs
    """
    rec = []
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloaders["train"]):
            model.train()  # put model to training mode
            x = x.to(device=device).float()
            y = y.to(device=device).long()
            y_ = model(x)
            loss = F.cross_entropy(y_, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with respect to each parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients computed by the backwards pass.
            optimizer.step()

        logging.info('epoche %d, loss = %f' % (e, loss.item()))
        train_acc = check_accuracy(device, dataloaders["train"], model, "train")
        test_acc = check_accuracy(device, dataloaders["val"], model, "val")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        rec.append((loss.item(), test_acc))
        
     
    
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model(save_dir = "fine_grained_classification/Embedding_Label_Structures_for_Fine_Grained_Feature_Representation/model_checkpoint", whole_model = False, file_name = task_name, model = model)
    return model, best_acc, rec

def save_model(save_dir, whole_model, file_name = None, model = None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))
    if model:
        if whole_model:
            torch.save(model, save_path + ".pkl")
        else:
            torch.save(model.state_dict(), save_path + ".pkl")
    else:
        logging.info("check point not saved, best_model is None")

def load_model(path, whole_model, model = None):
    if whole_model:
        model = torch.load(path)
    else:
        model.load_state_dict(torch.load(path))
    return model


def display_one_batch(image_dataset, dataloader):
    def imshow(inp, title=None, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(mean)
        std = np.array(std)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(100)
        
    class_names = image_dataset.classes
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))
    print(input, classes)
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# configuration
task_name = "test_run2"
model_name = "1dcnn"
optimizer_name = "Adam"
lr = 0.001
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
epochs = 100
logging.info(
    """{}:
    - model name: {}
    - optimizer: {}
    - learning rate: {}
    - device : {}
    - epochs: {}
 """.format(
        task_name, 
        model_name, 
        optimizer_name, 
        lr, 
        device, 
        epochs)
)

if __name__ == "__main__":
    model = cnn.cnn()
    # model.initialize()

    # get the param to update
    params_to_update = []
    for name, param in model.named_parameters():
        param.requires_grad = True
        params_to_update.append(param)
    


    # optimizer
    optimizer = getattr(optim, optimizer_name)(params_to_update, lr=lr)
    
    # dataLoaders
    dataLoaders = dataLoader.get_dataLoaders()
    
    # train and test

    best_model, best_acc, rec = train(model = model, 
                                    optimizer = optimizer, 
                                    dataloaders = dataLoaders, 
                                    device = device, 
                                    epochs = epochs
                                    )
    json.dump(open("rec_adam_no_L2.json", "w"), rec)
    
    
    

