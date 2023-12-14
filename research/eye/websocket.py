import asyncio
from websockets import serve
import numpy as np
import matplotlib.pyplot as plt
import json


async def handler(websocket):
    fig, ax = plt.subplots()
    eye_move_up_down, eye_move_left_right, blink_speed, blink_strength = [], [], [], []
    count = 0

    while True:
        count = + 1
        msg = await websocket.recv()
        # print(msg)
        obj = (json.loads(msg))
        eye_move_up_down.append(obj['eyeMoveUp'] - obj['eyeMoveDown'])
        eye_move_left_right.append(obj['eyeMoveLeft'] - obj['eyeMoveRight'])
        blink_speed.append(obj['blinkSpeed'])
        blink_strength.append(obj['blinkStrength'])

        if len(eye_move_up_down) > 100:
            del eye_move_up_down[0]
        if len(eye_move_left_right) > 100:
            del eye_move_left_right[0]
        if len(blink_speed) > 100:
            del blink_speed[0]
        if len(blink_strength) > 100:
            del blink_strength[0]

        x = range(len(eye_move_up_down))

        line_eye_move_up_down, = ax.plot(
            x, eye_move_up_down, color='C1', label='視線上下')
        line_eye_move_left_right, = ax.plot(
            x, eye_move_left_right, color='C2', label='視線左右')
        line_blink_speed, = ax.plot(x, blink_speed, color='C3', label='瞬き速度')
        line_blink_strength, = ax.plot(
            x, blink_strength, color='C4', label='瞬き強度')

        plt.pause(0.001)

        line_eye_move_up_down.remove()
        line_eye_move_left_right.remove()
        line_blink_speed.remove()
        line_blink_strength.remove()


async def start_server():
    print('Start')
    await serve(handler, '0.0.0.0', 19999)


if __name__ == "__main__":

    asyncio.get_event_loop().run_until_complete(start_server())
    asyncio.get_event_loop().run_forever()
