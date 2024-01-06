import os
import json
import yaml
import logging
import traceback
from pathlib import Path
from app import train
from dotenv import dotenv_values


envs = dotenv_values(".env.public")

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")

app_log = logging.getLogger(__name__)
    
async def handler(websocket):
    try:
        async for message in websocket:
            if isinstance(message, str):
                if message == "done":
                    await train.train(message, websocket)
                    return
                with open(TRAIN_PARAMS, "w") as f:
                    yaml.dump(json.loads(message), f)
                await websocket.send("200")
            else:
                await train.train(message, websocket)
    except Exception as err:
        app_log.error(err)
        if envs["MODE"] == "dev":
            traceback.print_tb(err.__traceback__)
        await websocket.send("500")
    