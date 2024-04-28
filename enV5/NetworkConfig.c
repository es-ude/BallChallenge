
#include "Network.h"
#include "MqttBroker.h"

networkCredentials_t networkCredentials = {
    .ssid = "SSID",
    .password = "PASSWORD"
};
mqttBrokerHost_t mqttHost = {
    .ip = "0.0.0.0",
    .port = "1883",
    .userID = "",
    .password = ""
};
