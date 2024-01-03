import os
import json
import yaml
import logging
import traceback
from pathlib import Path
from app import model
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PARAMS = os.path.join(BASE_DIR, "params/process_model.yml")

app_log = logging.getLogger(__name__)
    
async def handler(websocket):
    try:
        async for message in websocket:
            if isinstance(message, str):
                with open(TRAIN_PARAMS, "w") as f:
                    yaml.dump(json.loads(message), f)
                await websocket.send("200")
            else:
                await model.train(message, websocket)
    except Exception as err:
        app_log.error(err)
        if os.getenv("MODE") == "dev":
            traceback.print_tb(err.__traceback__)
        await websocket.send("500")
    