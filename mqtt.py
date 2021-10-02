import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    print(f'Connected with result code {rc}')
    client.subscribe('hello/#')


def on_message(client, userdata, msg):
    print(f'{msg.topic} {msg.payload.decode("utf-8")}')


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print('Connecting')
client.connect('localhost', 1883, 60)
client.loop_forever()
