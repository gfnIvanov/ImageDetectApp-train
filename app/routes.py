import os
import json
import yaml
import logging
import traceback
from pathlib import Path
from app import model
from dotenv import dotenv_values


envs = dotenv_values(".env.public")

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")
MODEL_PATH = os.path.join(BASE_DIR, "model/model.pth")
OPTIM_PATH = os.path.join(BASE_DIR, "model/optim.tar")

app_log = logging.getLogger(__name__)
    
async def handler(websocket):
    try:
        async for message in websocket:
            if isinstance(message, str):
                if message == "done":
                    await model.train(message, websocket)
                    return
                with open(TRAIN_PARAMS, "w") as f:
                    yaml.dump(json.loads(message), f)
                if os.path.exists(MODEL_PATH) and os.path.exists(OPTIM_PATH):
                    os.remove(MODEL_PATH)
                    os.remove(OPTIM_PATH)
                await websocket.send("200")
            else:
                await model.train(message, websocket)
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        await websocket.send("500")
    