import torchvision.transforms as transforms
import torchvision
import torch
import pickle
import subprocess
import os 

transform_train =  transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

def getTrainset(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, PIN_MEMORY=False):
    trainset = torchvision.datasets.CIFAR10(
        root='../Stack/Data',
        train=True,
        download=True,
        transform=transform_train
    )

    return torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NOM_WORKERS_TRAIN,
        pin_memory=False
    )

def generateJobs(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, SAVE_PATH):
    trainloader = getTrainset(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, PIN_MEMORY=True)

    for i, (image, label) in enumerate(trainloader):
        data = {
            "name": f"data_{i}",
            "image": image,
            "label": label
        }

        if os.path.exists(f'{SAVE_PATH}data_{i}.pkl'):
            os.remove(f'{SAVE_PATH}data_{i}.pkl')

        with open(f'{SAVE_PATH}data_{i}.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def generateJobs_synk(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, SAVE_PATH):
    cmd = 'python -c "from generateJobs import generateJobs; generateJobs({}, {}, \'{}\')"'.format(
        BATCH_SIZE_TRAIN,
        NOM_WORKERS_TRAIN,
        SAVE_PATH
    )

    subprocess.run(cmd, shell=True)

def generateJobs_asynk(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, SAVE_PATH):
    cmd = 'python -c "from generateJobs import generateJobs; generateJobs({}, {}, \'{}\')" &'.format(
        BATCH_SIZE_TRAIN,
        NOM_WORKERS_TRAIN,
        SAVE_PATH
    )

    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    JOBS_PATH = "../Stack/JobStack2/"
    BATCH_SIZE_TRAIN = 128
    NOM_WORKERS_TRAIN = 4

    print("ok")
    generateJobs_synk(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, JOBS_PATH)
    print("ok")
