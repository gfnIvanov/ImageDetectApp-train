import os
import pickle
import yaml
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import traceback
from pathlib import Path
from dotenv import dotenv_values


envs = {
    **dotenv_values(".env.public"),
    **dotenv_values(".env.secret"),
}

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")
MODEL_PATH = os.path.join(BASE_DIR, "model/model.pth")
OPTIM_PATH = os.path.join(BASE_DIR, "model/optim.tar")

with open(TRAIN_PARAMS) as f:
    params = yaml.safe_load(f)

app_log = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

async def train(message, websocket):
    try:
        if isinstance(message, str):            
            session = boto3.session.Session()

            s3 = session.client(
                service_name="s3",
                aws_access_key_id=envs["aws_access_key_id"],
                aws_secret_access_key=envs["aws_secret_access_key"],
                endpoint_url="https://storage.yandexcloud.net"
            )

            s3.upload_file(MODEL_PATH, envs["BUCKET"], "model.pth")
            await websocket.send("200")
            return
        
        inputs = torch.as_tensor(pickle.loads(message), dtype=torch.float32)

        NET = Net()

        if params["optim"] == "SDG":
            optimizer = optim.SGD(NET.parameters(), lr=float(params["learn-rate"]), momentum=0.9)
        else:
            optimizer = optim.Adam(NET.parameters(), lr=float(params["learn-rate"]), momentum=0.9) 

        if os.path.exists(MODEL_PATH) and os.path.exists(OPTIM_PATH):
            NET.load_state_dict(torch.load(MODEL_PATH))    
            optimizer.load_state_dict(torch.load(OPTIM_PATH))   
            NET.train()

        optimizer.zero_grad()

        outputs = NET(inputs)

        optimizer.step()

        torch.save(NET.state_dict(), MODEL_PATH)
        torch.save(optimizer.state_dict(), OPTIM_PATH)

        await websocket.send(pickle.dumps(outputs))
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        await websocket.send("500")