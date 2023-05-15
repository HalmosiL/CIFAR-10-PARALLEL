import requests
import pickle
import os
import time
import shutil
import random

import torchvision.transforms as transforms
import torchvision
import torch


DEVICE = 0

transform_test =  transforms.Compose([
    transforms.ToTensor(),
])

def getTestset(BATCH_SIZE_TEST, NOM_WORKERS_TEST, PIN_MEMORY=False):
    testset = torchvision.datasets.CIFAR10(
        root='/home/developer/Desktop/CIFAR-10-Paralel/Stack/Data',
        train=False,
        download=True,
        transform=transform_test
    )

    return torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=NOM_WORKERS_TEST,
        pin_memory=PIN_MEMORY
    )

class ClientDataset:
    def __init__(self, url):
        self.url = url
        self.ID = 0

    def getStaus(self):
        resp = requests.get(url=self.url + "/getStatus")
        return resp.json()

    def timeToWait(self, start, end):
        return random.uniform(start, end)

    def __getItem__(self):
        status = self.getStaus()['status']

        while(status == "Wait"):
            time.sleep(self.timeToWait(0.05, 0.15))
            status = self.getStaus()['status']

        if(status == 'Get data'):
            file_name = "data_" + str(self.ID) + ".pkl"

            r = requests.get(self.url + "/getData", stream=True)

            while(r.status_code == 404 and status != "Epoch Finished"):
                time.sleep(self.timeToWait(0.05, 0.15))
                status = self.getStaus()['status']

                if(status == 'Get data'):
                    r = requests.get(self.url + "/getData", stream=True)

            if(status != "Epoch Finished"):
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

                data = pickle.load(open(file_name, 'rb'))
                os.remove(file_name)

                self.ID += 1
                return data, "ok"

        return None, "Epoch Finished"

if __name__ == "__main__":
    clientDataset = ClientDataset("http://127.0.0.1:8000")

    start_time = time.time()
    cout = 0

    for i in range(10000):
        _, status = clientDataset.__getItem__()

        if(status == "Epoch Finished"):
            print(time.time() - start_time)
            print(cout)
            start_time = time.time()
            cout = 0
        else:
            cout += 1