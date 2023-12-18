from paho.mqtt import client as mqtt


class MqttClient:
    topics: dict[str, callable]
    client: mqtt.Client

    def __init__(self):
        self.topics = dict()
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        print("Connected to Broker")

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        print(f"Subscribed: {mid} with QOS: {granted_qos}")

    def _on_message(self, client, userdata, msg):
        print(f"Received \"{msg.topic}\" : {msg.payload}")
        self.topics.get(msg.topic)()

    def _on_disconnect(self, client, userdata, rc):
        print("Disconnected from Broker")

    def subscribe(self, topic: str, subscription_callback: callable):
        if self.topics.get(topic) is not None:
            raise Exception(f"Topic {topic} already exists")
        else:
            self.client.subscribe(topic)
            self.topics.update({topic: subscription_callback})

    def start(self, broker: str = "localhost", port: int = 1883):
        self.client.connect(host=broker, port=port)
        self.client.loop_start()

    def stop(self):
        self.client.disconnect()
        self.client.loop_stop()
