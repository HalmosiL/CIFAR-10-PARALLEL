import os
import pickle
import glob 

from sanic import Sanic, response, exceptions
from generateJobs import generateJobs_asynk, generateJobs_synk,getTrainset

EPOCH_COUNTER = 0
TIME_STAMP = 0

BATCH_SIZE_TRAIN = 128
NOM_WORKERS_TRAIN = 3

TRAIN_LOADER = getTrainset(BATCH_SIZE_TRAIN, NOM_WORKERS_TRAIN, PIN_MEMORY=False)

JOBS_PATH_1 = "../Stack/JobStack1/"
JOBS_PATH_2 = "../Stack/JobStack2/"

JOBS_PATH = JOBS_PATH_1

DATA_PATH = "../Stack/DataStack/"

MODELS_PATH = "../Stack/ModelCache/"
MODELS_NAMES = glob.glob(MODELS_PATH + "*.pt")

JOB_LIST = []
WORK_LIST = []
DATA_LIST = []

Finished_EPOCH = False

file_names = glob.glob(JOBS_PATH + "*")
file_names.sort()

JOB_LIST = [{"id":i, "name": name.split("/")[-1].split(".")[0], "path": name, "time-stamp": None} for i, name in enumerate(file_names)]
app = Sanic(name="LoadeBalancer")

def serachIndexByName(Data, name):
    for i in range(len(Data)):
        if(Data[i]["name"] == name):
            return i 

    return -1

def getNewModell():
    global MODELS_NAMES

    MODELS_NAMES = glob.glob(MODELS_PATH + "*.pt")
    
    max_index = 0
    max_index_name = ""

    for i in range(len(MODELS_NAMES)):
        value = int(MODELS_NAMES[i].split("_")[-1].split(".")[0])

        if(max_index < value):
            max_index = value
            max_index_name = MODELS_NAMES[i]

    return max_index, max_index_name

def regenerateData():
    global JOB_LIST
    global WORK_LIST
    global DATA_LIST

    JOB_LIST = []
    WORK_LIST = []

    file_names = glob.glob(JOBS_PATH + "*")
    file_names.sort()

    JOB_LIST = [{"id":i, "name": name.split("/")[-1].split(".")[0], "path": name, "model-id": getNewModell()[0]} for i, name in enumerate(file_names)]

@app.route('/getModelID')
async def getModelID(request):
    return response.text(str(getNewModell()[0]))

@app.route('/getModel')
async def getModel(request):
    print("Download Model")
    return await response.file(getNewModell()[1])

@app.route('/getJob')
async def getJob(request):
    global TIME_STAMP
    global JOB_LIST
    global WORK_LIST
    global DATA_LIST
    global Finished_EPOCH
    global EPOCH_COUNTER

    if(Finished_EPOCH or len(DATA_LIST) > 10):
        return response.text("Wait")

    if(len(JOB_LIST) == 0):
        if(len(WORK_LIST) == 0 and len(DATA_LIST) == 0 and not Finished_EPOCH):
            Finished_EPOCH = True

            if(EPOCH_COUNTER % 2 == 0):
                JOBS_PATH = JOBS_PATH_1
                regenerateData()

                generateJobs_asynk(
                    BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
                    NOM_WORKERS_TRAIN=NOM_WORKERS_TRAIN,
                    SAVE_PATH=JOBS_PATH_2
                )

                EPOCH_COUNTER += 1
            else:
                JOBS_PATH = JOBS_PATH_2
                regenerateData()

                generateJobs_asynk(
                    BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
                    NOM_WORKERS_TRAIN=NOM_WORKERS_TRAIN,
                    SAVE_PATH=JOBS_PATH_1
                )

                EPOCH_COUNTER += 1

            return response.text("Finished")
        else:
            return response.text("Wait")
    else:
        job = JOB_LIST[0]

        pickled_data = pickle.load(open(job["path"], 'rb'))

        pickled_data = {
            'name': pickled_data['name'],
            'image': pickled_data['image'],
            'label': pickled_data['label'],
            'model-id': getNewModell()[0]
        }

        pickled_data = pickle.dumps(pickled_data)

        res = response.raw(
            pickled_data,
            content_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={job['path'].split('/')[-1:][0]}"}
        )

        WORK_LIST.append(job)
        JOB_LIST.pop(0)

        return res

@app.route('/uploadJob', methods=['POST'])
async def upload(request):
    global TIME_STAMP

    uploaded_file = request.files.get('file')
    file_path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(file_path, 'wb') as f: f.write(uploaded_file.body)
    with open(file_path, 'rb') as f: obj = pickle.load(f)

    if obj["model-id"] == getNewModell()[0]:
        with open(f'{DATA_PATH}{obj["name"]}.pkl', 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        os.remove(file_path)

        TIME_STAMP += 1

        index = serachIndexByName(WORK_LIST, obj["name"])
        DATA_LIST.append(obj["name"])

        if(index != -1):
            WORK_LIST.pop(index)

        return response.json({'message': 'File uploaded successfully'})
    else:
        index = serachIndexByName(WORK_LIST, obj["name"])
        JOB_LIST.insert(0, WORK_LIST[index])

        if(index != -1): WORK_LIST.pop(index)

        return response.json({'message': 'File not uploaded'})

@app.route('/getStatus')
async def getStatus(request):
    global Finished_EPOCH

    if(Finished_EPOCH and len(DATA_LIST) == 0):
        Finished_EPOCH = False
        return response.json({'status': 'Epoch Finished'})

    if(not Finished_EPOCH and len(DATA_LIST) == 0):
        return response.json({'status': 'Wait'})

    return response.json({'status': "Get data"})

@app.route('/getData')
async def getData(request):
    pickled_data = None
    name = DATA_LIST[0]

    DATA_LIST.pop(0)

    if(len(DATA_LIST) == 0):
        raise exceptions.NotFound(f"Could not find any more data")

    with open(DATA_PATH + DATA_LIST[0] + ".pkl", 'rb') as f:
        pickled_data = pickle.load(f)
        pickled_data = pickle.dumps(pickled_data)

        res = response.raw(
            pickled_data,
            content_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={name + '.pkl'}"}
        )

        return res

def clearDataSet():
    data_set = glob.glob(DATA_PATH + "*.pkl")

    for i in data_set:
        os.remove(i)

if __name__ == "__main__":
    generateJobs_asynk(
        BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
        NOM_WORKERS_TRAIN=NOM_WORKERS_TRAIN,
        SAVE_PATH=JOBS_PATH_2
    )

    generateJobs_synk(
        BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
        NOM_WORKERS_TRAIN=NOM_WORKERS_TRAIN,
        SAVE_PATH=JOBS_PATH_1
    )

    clearDataSet()
    app.run(host="127.0.0.1", port=8000)