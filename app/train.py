import os
import pickle
import yaml
import boto3
import torch
import torch.optim as optim
import logging
import traceback
from pathlib import Path
from app import model
from dotenv import dotenv_values


envs = {
    **dotenv_values(".env.public"),
    **dotenv_values(".env.secret"),
}

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")
NET = model.Net()

with open(TRAIN_PARAMS) as f:
    params = yaml.safe_load(f)

app_log = logging.getLogger(__name__)

app_log.info(envs)

async def train(message, websocket):
    try:
        if isinstance(message, str):
            model_path = os.path.join(BASE_DIR, "model/model.pth")
            torch.save(NET.state_dict(), model_path)

            session = boto3.session.Session()

            s3 = session.client(
                service_name="s3",
                aws_access_key_id=envs["aws_access_key_id"],
                aws_secret_access_key=envs["aws_secret_access_key"],
                endpoint_url="https://storage.yandexcloud.net"
            )

            s3.upload_file(model_path, envs["BUCKET"], "model.pth")
            await websocket.send("200")
            return
        
        inputs = pickle.loads(message)

        if params["optim"] == "SDG":
            optimizer = optim.SGD(NET.parameters(), lr=float(params["learn-rate"]), momentum=0.9)
        else:
            optimizer = optim.Adam(NET.parameters(), lr=float(params["learn-rate"]), momentum=0.9) 

        optimizer.zero_grad()

        outputs = NET(inputs)
        await websocket.send(pickle.dumps(outputs))
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        await websocket.send("500")