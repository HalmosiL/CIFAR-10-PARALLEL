import torch
import json
import sys
import tqdm
import os
import sys

import glob
import wandb

import torch.optim as optim
import torch.nn as nn

sys.path.insert(1, '../')

from utils.Dataset import getTestset, ClientDataset
from utils.Model import getModel

os.environ["WANDB_RUN_GROUP"] = "ADVERSARIAL-PARALEL"

def Train(model, optimizer, trainloader, criterion, scheduler, step):
    global MODEL_ID
    global CASH_PERIOD

    model = model.train()

    correct = 0
    total = 0
    loss_log = 0

    len_ = 0
    data, status = trainloader.__getItem__()

    while(status == "ok"):
        inputs = data['image']
        labels = data['label']

        inputs = inputs.to(CONFIG["DEVICE"])
        labels = labels.to(CONFIG["DEVICE"])

        optimizer.zero_grad()

        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        data, status = trainloader.__getItem__()

        len_ += 1
        MODEL_ID += 1

        if(MODEL_ID % CASH_PERIOD == 0):
            torch.save(model.state_dict(), CASH_PATH + f"/model_{MODEL_ID}.pt")

        loss_log += loss.item()

    loss_log = loss_log / len_

    if scheduler is not None:
        scheduler.step()

    wandb.log({"train_acc": 100 * correct / total, "train_loss": loss_log}, step=step)

    return model

def Test(model, testloader, criterion, step):
    correct = 0
    total = 0
    loss = 0

    model = model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images = images.to(CONFIG["DEVICE"])
            labels = labels.to(CONFIG["DEVICE"])

            outputs = model(images)

            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss = loss / testloader.__len__()

    wandb.log({"val_acc": 100 * correct / total, "val_loss": loss}, step=step)

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

def schale(epoch):
    if(epoch <= 100):
        return 1
    
    if(epoch <= 150):
        return 0.1

    return 0.01

def clearModelCache(model_list):
    for m in model_list:
        os.remove(m)

def sort_models(val):
    return int(val.split("/")[-1].split("_")[-1][:-3])

CONFIG = json.load(open(sys.argv[1]))

NOM_WORKERS_TRAIN = CONFIG["NOM_WORKERS_TRAIN"]
NOM_WORKERS_TEST = CONFIG["NOM_WORKERS_TEST"]

MODEL_ID = 1
CASH_PERIOD = 100

NAME = CONFIG["NAME"] + "_MaxEpoch_" + str(CONFIG["EPOCHS"]) + "_BatchSize_" + str(CONFIG["BATCH_SIZE_TRAIN"])
SAVE_PATH = "../Models/" + NAME
CASH_PATH = "../../Stack/ModelCache/"

model_list = glob.glob(CASH_PATH + "*.pt")
model_list.sort(key=sort_models)
clearModelCache(model_list)

wandb.init(
    project="CIFAR-10-Adversarial-Traning",
    group="Normal",
    job_type="Train",
    config={
    "learning_rate": 0.1,
    "architecture": "RESNET18",
    "dataset": "CIFAR-10",
    "epochs": CONFIG["EPOCHS"],
    },

    name=NAME
)

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

model = getModel().to(CONFIG["DEVICE"])
torch.save(model.state_dict(), CASH_PATH + f"/model_1.pt")

clientDataset = ClientDataset("http://127.0.0.1:8000")
testloader = getTestset(CONFIG["BATCH_SIZE_TEST"], NOM_WORKERS_TEST, PIN_MEMORY=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[schale])

for epoch in tqdm.tqdm(range(CONFIG["EPOCHS"])):
    model = Train(model, optimizer, clientDataset, criterion, scheduler, epoch + 1)

    model_list = glob.glob(CASH_PATH + "*.pt")
    model_list.sort()
    clearModelCache(model_list[:-2])
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        Test(model, testloader, criterion, epoch + 1)
    
    if(epoch % CONFIG["ModelSavePeriod"] == 0):
        torch.save(model.state_dict(), SAVE_PATH + f"/model_{epoch}.pt")

print('Finished Training')
print('Save model:')

torch.save(model.state_dict(), SAVE_PATH + "/model_fin.pt")

