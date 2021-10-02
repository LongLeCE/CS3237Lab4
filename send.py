import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image
import json
from os import listdir
from os.path import join

GROUP = 99
HOSTNAME = 'localhost'
PATH = 'samples'


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print('Connected')
        client.subscribe(f'Group_{GROUP}/IMAGE/predict')
    else:
        print(f'Failed to connect. Error code: {rc}.')


def on_message(client, userdata, msg):
    print('Received message from server.')
    resp_dict = json.loads(msg.payload)
    print(f'Filename: {resp_dict["filename"]}, Prediction: {resp_dict["prediction"]}, Score: {resp_dict["score"]:3.4f}')


def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client


def load_image(filename):
    img = Image.open(filename)
    img = img.resize((249, 249))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final


def send_image(client, filename):
    img = load_image(filename)
    img_list = img.tolist()
    send_dict = {
        'filename': filename,
        'data': img_list
    }
    client.publish(f'Group_{GROUP}/IMAGE/classify', json.dumps(send_dict))


def main():
    client = setup(HOSTNAME)
    print('Sending data.')
    # Loop through all images in PATH
    for file in listdir(PATH):
        # Get path to image
        file_path = join(PATH, file)
        print(f'Sending image {file_path}')
        # Send image classification request
        send_image(client, file_path)
    print('Done. Waiting for results.')
    while True:
        pass


if __name__ == '__main__':
    main()
