import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")

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
        inputs = pickle.loads(message)
        net = Net()

        if params["optim"] == "SDG":
            optimizer = optim.SGD(net.parameters(), lr=float(params["learn-rate"]), momentum=0.9)
        else:
            optimizer = optim.Adam(net.parameters(), lr=float(params["learn-rate"]), momentum=0.9) 

        optimizer.zero_grad()

        outputs = net(inputs)
        await websocket.send(pickle.dumps(outputs))
    except Exception as err:
        app_log.error(err)
        if os.getenv("MODE") == "dev":
            traceback.print_tb(err.__traceback__)
        return { "status": 500 }