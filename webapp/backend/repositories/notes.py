"""Access to elastic-AI Nodes."""

from time import sleep

from elasticai.mqtt.client import MQTTClient
from elasticai.protocol.base import DeviceState, DeviceType
from elasticai.protocol.base import Protocol as EAIP
from elasticai.protocol.data_requester import DataRequester


class Nodes:
    """List of available enV5's."""

    __availableNodes: set[str]
    __mqtt_client: MQTTClient
    __mqtt_broker_ip: str
    __mqtt_broker_port: int
    __eaip: EAIP

    def __init__(
        self,
        domain: str = "eaip://ballchallenge.ies",
        host: str = "localhost",
        port: int = 1883,
        id: str = "bc-webapp",
        homepage: str = "http://127.0.0.1:8081/",
    ) -> None:
        """Initialise instance of class.

        Args:
            domain (str): base domain for the protocol communication
                          default: "eaip://ballchallenge.ies"
            host (str): host of the MQTT Broker
                        default: "localhost"
            port (int): port of the MQTT Broker
                        default: 1883
            id (str): client id for the protocol:
                      default: "bc-webapp"
            homepage (str): homepage to view the ball challenge app
                           default: "http://127.0.0.1:8081/",

        Returns:
            None
        """
        self.__mqtt_broker_ip: str = host
        self.__mqtt_broker_port: int = port

        self.__mqtt_client = MQTTClient()
        self.__eaip = EAIP(handler=self.__mqtt_client, device_id=id, base_url=domain)
        lwt = self.__eaip.get_lwt()

        self.__mqtt_client.set_lwt(topic=lwt["topic"], payload=lwt["message"])
        self.__mqtt_client.connect(host, port)

        self.__eaip.publish_status(DeviceState.ONLINE, {"homepage": homepage})
        self.__availableNodes = set()
        self.__eaip.subscribe_status("+", self.__status_handler)

    def __status_handler(self, topic: str, msg: str) -> None:
        status_dict = self.__eaip.parse_status(msg)

        if not status_dict["TYPE"] == DeviceType.NODE.value:
            return

        if status_dict["STATE"] == DeviceState.ONLINE.value:
            self.__availableNodes.add(status_dict["ID"])
        else:
            self.__availableNodes.remove(status_dict["ID"])

    def find_by_name(self, name: str) -> DataRequester | None:
        """Get communication endpoint for enV5."""
        if name in self.__availableNodes:
            request_client: MQTTClient = MQTTClient()
            request_client.connect(self.__mqtt_broker_ip, self.__mqtt_broker_port)
            data_requester: DataRequester = DataRequester(
                target_device=name,
                target_data_id="acceleration",
                client=request_client,
                client_device_id=self.__eaip.get_device_id(),
                client_type=self.__eaip.get_device_type(),
                base_url=self.__eaip.get_base_url(),
            )

            sleep(
                1
            )  # required to assure status handling is done before calling start or stop
            return data_requester
        else:
            return None

    def find_all(self) -> list[str]:
        """List of all nodes."""
        return list(self.__availableNodes)


if __name__ == "__main__":
    nodes: Nodes = Nodes("eaip://ballchallenge.ies/", "localhost", 1883)
    while len(nodes.find_all()) == 0:
        pass
    print(nodes.find_all())
    print("REACHED")
