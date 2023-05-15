import requests
import pickle
import time
import shutil
import torch.nn as nn

from utils.Model import loadModel
from utils.Adversarial import PGD

import sys
import json
import os
import glob

CONFIG = json.load(open(sys.argv[1]))

MODEL_ID = None

DEVICE = CONFIG["DEVICE"]
HOST = CONFIG["DATA-LOADER-HOST"]

def clearModelCache(model_list):
    if(len(model_list) > 2):
        for m in model_list:
            try:
                os.remove(m)
            except:
                print("Can't delete:", m)

def getModelID(url):
    response = requests.get(url)
    return int(response.content.decode("utf-8"))

def getModel(url, MODEL):
    global MODEL_ID

    local_filename = url.split('/')[-1]
    model = None

    try:
        with requests.get(url, stream=True) as r:
            with open(f"./ModelCache/model_{MODEL_ID}.pt", 'wb') as f:
                shutil.copyfileobj(r.raw, f)
            
        model = loadModel(f"../Stack/ModelCache/model_{MODEL_ID}.pt")
        model_list = glob.glob("./ModelCache/*.pt")
        model_list.sort(key=sort_models)
        clearModelCache(model_list[:-2])
    except Exception as e:
        print(e)
        print("Model read error:", MODEL_ID)
        return MODEL

    MODEL_ID += 1

    return model

def load_pickle_file(url):
    response = requests.get(url)

    if(response.content == b"Wait"):
        return "Wait"

    if(response.content == b"Finished"):
        return "Finished"

    file_contents = response.content

    try:
        obj = pickle.loads(file_contents)
        return obj
    except:
        return "Wait"

def upload_pickle_file(url, obj):
    files = {'file': ('file.pkl', pickle.dumps(obj))}
    response = requests.post(url, files=files)
    return response.status_code

def sort_models(val):
    return int(val.split("/")[-1].split("_")[-1][:-3])

if __name__ == "__main__":
    MODEL = None

    model_list = glob.glob("./ModelCache/*.pt")
    model_list.sort(key=sort_models)
    clearModelCache(model_list)

    while True:
        obj = load_pickle_file(f"{HOST}/getJob")
        id_ = getModelID(f"{HOST}/getModelID")

        if MODEL_ID is not None:
            if(MODEL_ID < id_):
                MODEL_ID = id_
                MODEL = getModel(f"{HOST}/getModel", MODEL)
        else:
            MODEL_ID = id_
            MODEL = getModel(f"{HOST}/getModel", MODEL)

        if(MODEL is not None):
            MODEL = MODEL.to(DEVICE)

            if(obj != "Finished"):
                if(obj == "Wait" and MODEL is not None):
                    time.sleep(1)
                else:
                    obj['image'] = PGD(
                        obj['image'].to(DEVICE),
                        obj['label'].to(DEVICE),
                        MODEL,
                        epsilon=8/255,
                        stepSize=2/255,
                        lossFun=nn.CrossEntropyLoss(),
                        iterationNumber=10
                    )

                    upload_pickle_file(f"{HOST}/uploadJob", obj)
            else:
                print("Finished")
        else:
            print("Wait for the model")
