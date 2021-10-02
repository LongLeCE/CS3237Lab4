import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import paho.mqtt.client as mqtt
import numpy as np
import json
from time import sleep
from collections import deque

request_queue = deque()

classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

GROUP = 99
HOSTNAME = 'localhost'

MODEL_NAME = 'flowers.hd5'

session = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
with session.graph.as_default():
    set_session(session)
    model = load_model(MODEL_NAME)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print('Successfully connected to broker.')
        client.subscribe(f'Group_{GROUP}/IMAGE/classify')
    else:
        print(f'Connection failed with code: {rc}.')


def classify_flower(filename, data):
    print('Start classifying')
    # Set the current graph and session to the one with the model weights loaded, thus avoiding having to load the model again
    with session.graph.as_default():
        set_session(session)
        # Do prediction
        result = model.predict(data)
        # 'win' is the index of the maximum output value (highest probability)
        win = np.argmax(result).item()
        # 'score' is the output value (probability) at index 'win'
        score = result[0][win].item()
    print('Done.')
    # Return dictionary of results
    return {
        'filename': filename,
        'prediction': classes[win], # Map index to class name
        'score': score,
        'index': win
    }


def on_message(client, userdata, msg):
    request_queue.append((client, msg))


def process_request(client, msg):
    # Payload is in msg. We convert it back to a Python dictionary
    recv_dict = json.loads(msg.payload)

    # Recreate the data
    img_data = np.array(recv_dict['data'])
    result = classify_flower(recv_dict['filename'], img_data)

    print(f'Sending results: {result}')
    client.publish(f'Group_{GROUP}/IMAGE/predict', json.dumps(result))


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def main():
    setup(HOSTNAME)
    while True:
        if request_queue:
            client, msg = request_queue.popleft()
            process_request(client, msg)
        else:
            sleep(0.1)


if __name__ == '__main__':
    main()
