/*!
 * This implementation intends to provide a sample data collection.
 * It is intended to be used with the CSV Service as a backend (See README.adoc).
 */

#define SOURCE_FILE "DATA-COLLECT-APP"

// internal headers
#include "Adxl345b.h"
#include "Common.h"
#include "Esp.h"
#include "FreeRtosMutexWrapper.h"
#include "FreeRtosQueueWrapper.h"
#include "FreeRtosTaskWrapper.h"
#include "Gpio.h"
#include "MqttBroker.h"
#include "Network.h"
#include "Posting.h"
#include "Protocol.h"
#include "enV5HwController.h"

// pico-sdk headers
#include "hardware/i2c.h"
#include "hardware/watchdog.h"
#include "pico/bootrom.h"
#include "pico/stdlib.h"

// external headers
#include <string.h>

/* region VARIABLES/DEFINES */

#define EIP_BASE "eip://uni-due.de/es"
#define EIP_DEVICE_ID "dataCollect01"
status_t status = {.data = "g-value,timer"};

const uint8_t batchIntervalInSeconds = 3;
const uint16_t samplesPerSecond = 400;

typedef enum {
    DATA_VALUE,
} pubType_t;
typedef struct publishRequest {
    pubType_t pubType;
    char *topic;
    char *data;
} publishRequest_t;

queue_t receivedPosts;
queue_t batchRequest;
queue_t publishRequests;

mutex_t espOccupied;

/* endregion VARIABLES/DEFINES */

/* region HELPER FUNCTIONS */

static void initialize(void) {
    // check if we crash last time -> if true, reboot into boot rom mode
    if (watchdog_enable_caused_reboot()) {
        reset_usb_boot(0, 0);
    }

    env5HwInit();

    // initialize I/O for Debug purposes
    stdio_init_all();
//    while ((!stdio_usb_connected())) { /* wait for serial connection */
//    }

    espInit(); // initialize Wi-Fi chip
    networkTryToConnectToNetworkUntilSuccessful();
    mqttBrokerConnectToBrokerUntilSuccessful(EIP_BASE, EIP_DEVICE_ID);

    adxl345bErrorCode_t errorADXL = adxl345bInit(i2c1, ADXL345B_I2C_ALTERNATE_ADDRESS);
    i2c_set_baudrate(i2c1, 2000000);
    if (errorADXL == ADXL345B_NO_ERROR) {
        PRINT_DEBUG("Initialised ADXL345B.");
        adxl345bWriteConfigurationToSensor(ADXL345B_REGISTER_BW_RATE, ADXL345B_BW_RATE_400);
        adxl345bChangeMeasurementRange(ADXL345B_16G_RANGE);
    } else {
        PRINT_DEBUG("Initialise ADXL345B failed; adxl345b_ERROR: %02X", errorADXL);
        reset_usb_boot(0, 0);
    }
}

_Noreturn void watchdogTask(void) {
    protocolPublishData("test", "watchdog");
    watchdog_enable(10000, 1); // enables watchdog timer (10s)

    while (1) {
        watchdog_update();                  // watchdog update needs to be performed frequent
        freeRtosTaskWrapperTaskSleep(1000); // sleep for 1 second
    }
}

void deliver(posting_t posting) {
    freeRtosQueueWrapperPushFromInterrupt(receivedPosts, &posting);
}

_Noreturn void handleReceivedPostingsTask(void) {
    while (1) {
        posting_t post;
        if (freeRtosQueueWrapperPop(receivedPosts, &post)) {
            PRINT("Received Message: '%s' via topic '%s'", post.data, post.topic);
            if (NULL != strstr(post.topic, "/DO/MEASUREMENT")) {
                freeRtosQueueWrapperPush(batchRequest, NULL);
            }
            free(post.topic);
            free(post.data);
        }
        freeRtosTaskWrapperTaskSleep(500);
    }
}

_Noreturn void handlePublishTask(void) {
    publishAliveStatusMessageWithMandatoryAttributes(status);
    protocolSubscribeForCommand("MEASUREMENT", (subscriber_t){.deliver = deliver});

    while (1) {
        publishRequest_t request;
        if (freeRtosQueueWrapperPop(publishRequests, &request)) {
            PRINT("Publish request of type '%u' to topic '%s'", request.pubType, request.topic);
            switch (request.pubType) {
            case DATA_VALUE:
                freeRtosMutexWrapperLock(espOccupied);
                protocolPublishData(request.topic, request.data);
                freeRtosMutexWrapperUnlock(espOccupied);
                break;
            default:
                PRINT_DEBUG("type NOT valid!");
            }

            free(request.topic);
            free(request.data);
        }
        freeRtosTaskWrapperTaskSleep(500);
    }
}

static void showCountdown(void) {
    env5HwLedsAllOff();

    publishRequest_t pubRequest3 = {.pubType = DATA_VALUE, .topic = malloc(5), .data = malloc(2)};
    strcpy(pubRequest3.topic, "time");
    strcpy(pubRequest3.data, "3");
    freeRtosQueueWrapperPush(publishRequests, &pubRequest3);
    gpioSetPin(GPIO_LED0, GPIO_PIN_HIGH);
    freeRtosTaskWrapperTaskSleep(1000);

    publishRequest_t pubRequest2 = {.pubType = DATA_VALUE, .topic = malloc(5), .data = malloc(2)};
    strcpy(pubRequest2.topic, "time");
    strcpy(pubRequest2.data, "2");
    freeRtosQueueWrapperPush(publishRequests, &pubRequest2);
    gpioSetPin(GPIO_LED1, GPIO_PIN_HIGH);
    freeRtosTaskWrapperTaskSleep(1000);

    publishRequest_t pubRequest1 = {.pubType = DATA_VALUE, .topic = malloc(5), .data = malloc(2)};
    strcpy(pubRequest1.topic, "time");
    strcpy(pubRequest1.data, "1");
    freeRtosQueueWrapperPush(publishRequests, &pubRequest1);
    gpioSetPin(GPIO_LED2, GPIO_PIN_HIGH);
    freeRtosTaskWrapperTaskSleep(1000);

    env5HwLedsAllOff();
    freeRtosTaskWrapperTaskSleep(250);
    publishRequest_t pubRequest0 = {.pubType = DATA_VALUE, .topic = malloc(5), .data = malloc(2)};
    strcpy(pubRequest0.topic, "time");
    strcpy(pubRequest0.data, "0");
    freeRtosQueueWrapperPush(publishRequests, &pubRequest0);
}

static bool getSample(uint32_t *timeOfMeasurement, float *xAxis, float *yAxis, float *zAxis) {
    *timeOfMeasurement = time_us_32();

    adxl345bErrorCode_t errorCode = adxl345bReadMeasurements(xAxis, yAxis, zAxis);
    if (errorCode != ADXL345B_NO_ERROR) {
        PRINT_DEBUG("ERROR in Measuring G Value!");
        return false;
    }
    return true;
}

static char *appendSample(char *dest, float xAxis, float yAxis, float zAxis) {
    snprintf(dest, 15, "%13.9f,", xAxis);
    dest += 14;
    snprintf(dest, 15, "%13.9f,", yAxis);
    dest += 14;
    snprintf(dest, 15, "%13.9f\n", zAxis);
    dest += 14;
    return dest;
}

static char *collectSamples(void) {
    // axis: 3; char per value: 14; String Terminator: 1B
    char *data = malloc(batchIntervalInSeconds * samplesPerSecond * 3 * 14 + 1);
    char *nextEntryStart = data;
    uint16_t sampleCount = 0;
    uint32_t limit = time_us_32() + batchIntervalInSeconds * 1000000;
    uint32_t lastMeasurement = time_us_32();

    float xAxis, yAxis, zAxis;
    while (limit >= time_us_32() && sampleCount <= (samplesPerSecond * batchIntervalInSeconds)) {
        if (lastMeasurement + (1000000 / samplesPerSecond) < time_us_32()) {
            continue;
        }

        sleep_ms(1); // IMPORTANT: Has to be there! If deleted won't work!!!
        if (!getSample(&lastMeasurement, &xAxis, &yAxis, &zAxis)) {
            continue;
        }
        nextEntryStart = appendSample(nextEntryStart, xAxis, yAxis, zAxis);
        sampleCount++;
    }
    PRINT_DEBUG("GOT %u samples", sampleCount);

    return data;
}

static void publishMeasurements(char *data) {
    if (strlen(data) > 0) {
        char *topic = malloc(strlen("g-value") + 1);
        strcpy(topic, "g-value");
        publishRequest_t batchToPublish = {.pubType = DATA_VALUE, .topic = topic, .data = data};
        freeRtosQueueWrapperPush(publishRequests, &batchToPublish);
    } else {
        free(data);
    }
}

_Noreturn void recordMeasurementBatchTask(void) {
    while (1) {
        if (freeRtosQueueWrapperPop(batchRequest, NULL)) {
            showCountdown();
            char *data = collectSamples();
            publishMeasurements(data);
            env5HwLedsAllOn();
        }
        freeRtosTaskWrapperTaskSleep(500);
    }
}

/* endregion HELPER FUNCTIONS */

int main() {
    initialize();

    env5HwLedsAllOn();

    receivedPosts = freeRtosQueueWrapperCreate(10, sizeof(posting_t));
    batchRequest = freeRtosQueueWrapperCreate(5, 0);
    publishRequests = freeRtosQueueWrapperCreate(10, sizeof(publishRequest_t));

    espOccupied = freeRtosMutexWrapperCreate();

    freeRtosTaskWrapperRegisterTask(watchdogTask, "watchdog", 1,
                                    FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(handleReceivedPostingsTask, "receiver",
                                    1, FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(handlePublishTask, "sender", 2,
                                    FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(recordMeasurementBatchTask, "recorder", 2,
                                    FREERTOS_CORE_1);

    freeRtosTaskWrapperStartScheduler();
}
