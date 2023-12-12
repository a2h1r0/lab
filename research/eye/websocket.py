import asyncio
from websockets import serve


async def handler(websocket):
    while True:
        msg = await websocket.recv()
        print(msg)


async def start_server():
    print('Start')
    await serve(handler, '0.0.0.0', 19999)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(start_server())
    asyncio.get_event_loop().run_forever()
