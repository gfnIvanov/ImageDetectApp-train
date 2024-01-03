import os
import logging
import asyncio
import websockets
from app import routes
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level = logging.DEBUG,
                    filename = os.path.join(BASE_DIR, 'logs/app_log.log'),
                    filemode = "w",
                    format = "%(asctime)s - %(name)s[%(funcName)s(%(lineno)d)] - %(levelname)s - %(message)s")

async def main():
    async with websockets.serve(routes.handler, os.getenv("HOST"), os.getenv("PORT"), max_size = None):
        await asyncio.Future()

def run():
    asyncio.run(main())