/*
 * This implementation is intended to provide FPGA accelerated predictions.
 * It is intended to be used in combination with Monitor to:
 *   - Flash the FPGA to the device
 *   - Show the Results of the prediction
 *
 * When the FPGA is flashed successful,
 * a data recording is triggered and passed as input to the accelerator.
 * The result of the computation will be published to the topic
 * "eip://uni-due.de/es/<client_id>/DATA/accelerator/prediction".
 */

/* FIXME Concept:
 *   1. Receive Download Request
 *   2. Copy raw message to queue for message-Task
 *       a. Parse Message in message-Task
 *       b. Push Download Request into queue to network-Task
 *   5. Download binary to flash in network-Task
 *       a. Download binary to flash
 *       b. Push request for test execution into queue for test-Task
 *   8. Execute test in test-Task
 *       a. Get Test Data from Sensor
 *       b. Flash FPGA
 *       c. Send Batch to FPGA
 *       d. Perform prediction
 *       e. Push result to queue for network-Task
 *   9. Publish result in network-Task
 *
 *
 * test queue:
 * -----------
 * 5 Elements of: {uint32_t fpgaConfigAddress}
 *
 * network queue:
 * --------------
 * 5 Elements of: {enum requestType, uint8_t* request}
 * enum requestType = {DOWNLOAD, PUBLISH}
 * struct for download request: {char* url, uint32_t configSizeInBytes, uint16_t pageId}
 * struct for publish request: {char* topic, char* message}
 * cast request according to requestType!
 */

#define SOURCE_FILE "WRIST-APP"

// internal headers
#include "Adxl345b.h"
#include "Common.h"
#include "Esp.h"
#include "Flash.h"
#include "FpgaConfigurationHandler.h"
#include "FreeRtosQueueWrapper.h"
#include "FreeRtosTaskWrapper.h"
#include "MqttBroker.h"
#include "Network.h"
#include "Protocol.h"
#include "Spi.h"
#include "enV5HwController.h"

// pico-sdk headers
#include "hardware/i2c.h"
#include "hardware/spi.h"
#include "hardware/watchdog.h"
#include "pico/bootrom.h"
#include "pico/stdlib.h"

// external headers
#include <malloc.h>
#include <math.h>
#include <string.h>

/* region VARIABLES/DEFINES */

spi_t spiToFlash = {.spi = spi0, .baudrate = 5000000, .sckPin = 2, .mosiPin = 3, .misoPin = 0};
uint8_t flashChipSelectPin = 1;

status_t status = {
    .id = "dataCollect01", .type = "enV5", .state = STATUS_STATE_ONLINE, .data = "g-value,timer"};

bool subscribed = false;
char *twinID;
char *gValueDataBatch;

bool newBatchAvailable = false;
bool newBatchRequested = false;

const uint8_t batchIntervalInSeconds = 3;
const uint32_t batchIntervalInMicroSeconds = batchIntervalInSeconds * 1000000;
const uint16_t samplesPerSecond = 400;

typedef struct downloadRequest {
    char *url;
    size_t fileSizeInBytes;
    size_t startAddress;
} downloadRequest_t;
downloadRequest_t *downloadRequest = NULL;

/* endregion VARIABLES/DEFINES */

/* region PROTOTYPES */

static void initialize(void);

_Noreturn void watchdogTask(void);
_Noreturn void messageTask(void);
_Noreturn void networkTask(void);
_Noreturn void testTask(void);

void setTwinID(char *newTwinID);
void twinsIsOffline(posting_t posting);
void receiveDataStartRequest(posting_t posting);
void receiveDataStopRequest(__attribute__((unused)) posting_t posting);
void publishGValueBatch(char *dataID);
void receiveDownloadBinRequest(posting_t posting);

/* endregion HEADER */

int main() {
    initialize();

    freeRtosTaskWrapperRegisterTask(watchdogTask, "watchdogTask", 31, FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(networkTask, "networkTask", 1, FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(testTask, "testTask", 15, FREERTOS_CORE_1);
    freeRtosTaskWrapperStartScheduler();
}

/* region PROTOTYPE IMPLEMENTATIONS */

void receiveDownloadBinRequest(posting_t posting) {
    // FIXME:
    //   Copy message to queue to network Task
    //   Handle Download in network Task

    // get download request
    char *urlStart = strstr(posting.data, "URL:") + 4;
    char *urlEnd = strstr(urlStart, ";") - 1;
    size_t urlLength = urlEnd - urlStart + 1;
    char *url = malloc(urlLength);
    memcpy(url, urlStart, urlLength);
    url[urlLength - 1] = '\0';
    char *sizeStart = strstr(posting.data, "SIZE:") + 5;
    char *endSize = strstr(sizeStart, ";") - 1;
    size_t length = strtol(sizeStart, &endSize, 10);

    char *positionStart = strstr(posting.data, "POSITION:") + 9;
    char *positionEnd = strstr(positionStart, ";") - 1;
    size_t position = strtol(positionStart, &positionEnd, 10);

    downloadRequest = malloc(sizeof(downloadRequest_t));
    downloadRequest->url = url;
    downloadRequest->fileSizeInBytes = length;
    downloadRequest->startAddress = position;
}
void initialize(void) {
    // check if we crash last time -> reboot into boot rom mode
    if (watchdog_enable_caused_reboot()) {
        reset_usb_boot(0, 0);
    }

    env5HwInit();

    // init IO
    stdio_init_all();
    // waits for usb connection, REMOVE to continue without waiting for connection
    // while ((!stdio_usb_connected())) {}

    espInit(); // initialize Wi-Fi chip
    networkTryToConnectToNetworkUntilSuccessful();
    mqttBrokerConnectToBrokerUntilSuccessful("eip://uni-due.de/es", "enV5");

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

    flashInit(&spiToFlash, flashChipSelectPin);
    fpgaConfigurationHandlerInitialize();

    freeRtosQueueWrapperCreate();

    protocolSubscribeForCommand("FLASH", (subscriber_t){.deliver = receiveDownloadBinRequest});

    watchdog_enable(10000, 1); // enables watchdog timer (10s)
}

_Noreturn void watchdogTask(void) {
    while (true) {
        // watchdog update needs to be performed frequent, otherwise the device will crash
        watchdog_update();
        freeRtosTaskWrapperTaskSleep(1000);
    }
}

_Noreturn void networkTask(void) {}

_Noreturn void getGValueTask(void) {
    newBatchAvailable = false;
    uint16_t batchSize = 400;
    uint32_t interval = G_VALUE_BATCH_SECONDS * 1000000;

    gValueDataBatch = malloc(G_VALUE_BATCH_SECONDS * 11 * batchSize * 3 + 16);
    char *data = malloc(G_VALUE_BATCH_SECONDS * 11 * batchSize * 3 + 16);
    char timeBuffer[15];
    adxl345bWriteConfigurationToSensor(ADXL345B_REGISTER_BW_RATE, ADXL345B_BW_RATE_400);
    adxl345bChangeMeasurementRange(ADXL345B_16G_RANGE);

    uint32_t count;

    while (1) {
        if (!newBatchRequested) {
            freeRtosTaskWrapperTaskSleep(500);
            continue;
        }
        newBatchRequested = false;

        env5HwLedsInit();
        gpio_put(GPIO_LED0, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        gpio_put(GPIO_LED1, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        gpio_put(GPIO_LED2, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        env5HwLedsAllOff();
        freeRtosTaskWrapperTaskSleep(250);
        env5HwLedsAllOn();
        freeRtosTaskWrapperTaskSleep(250);
        env5HwLedsAllOff();

        snprintf(timeBuffer, sizeof(timeBuffer), "%lu", time_us_64() / 1000000);
        strcpy(data, timeBuffer);
        strcat(data, ",");
        count = 0;
        uint32_t currentTime = time_us_64();
        uint32_t startTime = time_us_64();
        while (startTime + interval >= currentTime) {
            currentTime = time_us_64();
            if (count >= batchSize * G_VALUE_BATCH_SECONDS) {
                continue;
            }
            float xAxis, yAxis, zAxis;
            adxl345bErrorCode_t errorCode = adxl345bReadMeasurements(&xAxis, &yAxis, &zAxis);
            if (errorCode != ADXL345B_NO_ERROR) {
                PRINT("ERROR in Measuring G Value!")
                continue;
            }

            char xBuffer[10];
            char yBuffer[10];
            char zBuffer[10];
            snprintf(xBuffer, sizeof(xBuffer), "%.10f", xAxis / 8);
            snprintf(yBuffer, sizeof(yBuffer), "%.10f", yAxis / 8);
            snprintf(zBuffer, sizeof(zBuffer), "%.10f", zAxis / 8);

            strcat(data, xBuffer);
            strcat(data, ",");
            strcat(data, yBuffer);
            strcat(data, ",");
            strcat(data, zBuffer);
            strcat(data, ";");
            count += 1;
        }
        if (count > 0) {
            PRINT_DEBUG("COUNT: %lu", count)
            newBatchAvailable = true;
            strcpy(gValueDataBatch, data);
        }
    }
}

_Noreturn void fpgaTask(void) {
    /* NOTE:
     *   1. add listener for download start command (MQTT)
     *      uart handle should only set flag -> download handled at task
     *   2. download data from server and stored to flash
     *   4. add listener for FPGA flashing command
     *   5. trigger flash of FPGA
     *      handled in UART interrupt
     */

    freeRtosTaskWrapperTaskSleep(5000);
    protocolSubscribeForCommand("FLASH", (subscriber_t){.deliver = receiveDownloadBinRequest});

    PRINT("FPGA Ready ...")

    while (1) {
        if (downloadRequest == NULL) {
            freeRtosTaskWrapperTaskSleep(1000);
            continue;
        }

        env5HwFpgaPowersOff();

        PRINT_DEBUG("Download: position in flash: %i, address: %s, size: %i",
                    downloadRequest->startAddress, downloadRequest->url,
                    downloadRequest->fileSizeInBytes)

        fpgaConfigurationHandlerError_t configError =
            fpgaConfigurationHandlerDownloadConfigurationViaHttp(downloadRequest->url,
                                                                 downloadRequest->fileSizeInBytes,
                                                                 downloadRequest->startAddress);

        // clean artifacts
        free(downloadRequest->url);
        free(downloadRequest);
        downloadRequest = NULL;
        PRINT("Download finished!")

        if (configError != FPGA_RECONFIG_NO_ERROR) {
            protocolPublishCommandResponse("FLASH", false);
            PRINT("ERASE ERROR")
        } else {
            freeRtosTaskWrapperTaskSleep(10);
            env5HwFpgaPowersOn();
            PRINT("FPGA reconfigured")
            protocolPublishCommandResponse("FLASH", true);
        }
    }
}

void receiveDataStartRequest(posting_t posting) {
    setTwinID(posting.data);
    subscribed = true;
    PRINT("Started requesting g-value")
}

void receiveDataStopRequest(__attribute__((unused)) posting_t posting) {
    subscribed = false;
    PRINT("Stopped requesting g-value")
}

void receiveSetSensorFrequency(posting_t posting) {
    int buf = strtol(posting.data, NULL, 10);
    PRINT("Setting Frequency to: %i", buf)
    uint8_t rate = (uint8_t)(log(buf) / log(2) + 4);
    if (rate > ADXL345B_BW_RATE_400) {
        PRINT("Frequency exceeds maximum allowed frequency")
        return;
    }
    if (adxl345bWriteConfigurationToSensor(ADXL345B_REGISTER_BW_RATE, rate) == ADXL345B_NO_ERROR) {
        PRINT("Set Frequency successfully")
    } else {
        PRINT("ERROR setting Frequency")
    }
}

_Noreturn void publishValueBatchesTask(void) {
    publishAliveStatusMessage("g-value");

    protocolSubscribeForCommand("setFrequency",
                                (subscriber_t){.deliver = receiveSetSensorFrequency});

    protocolSubscribeForDataStartRequest("g-value",
                                         (subscriber_t){.deliver = receiveDataStartRequest});
    protocolSubscribeForDataStopRequest("g-value",
                                        (subscriber_t){.deliver = receiveDataStopRequest});
    PRINT("Ready ...")

    uint64_t seconds;
    uint64_t lastPublished = 0;
    subscribed = false;
    while (true) {
        freeRtosTaskWrapperTaskSleep(100);
        if (subscribed) {
            seconds = (time_us_64()) / 1000000;
            if (seconds - lastPublished >= G_VALUE_PAUSE_SECONDS) {
                newBatchRequested = true;
                while (!newBatchAvailable) {
                    freeRtosTaskWrapperTaskSleep(100);
                }
                newBatchAvailable = false;
                lastPublished = (time_us_64()) / 1000000;
                publishGValueBatch("g-value");
                PRINT("Published G-Values (sec: %lu)", lastPublished)
            }
        }
    }
}

void setTwinID(char *newTwinID) {
    if (newTwinID == twinID) {
        return;
    }

    if (newTwinID != NULL) {
        free(twinID);
    }

    twinID = malloc(strlen(newTwinID) + 1);
    strcpy(twinID, newTwinID);
}

void twinsIsOffline(posting_t posting) {
    if (strstr(posting.data, STATUS_STATE_ONLINE) != NULL) {
        return;
    }
    PRINT("Twin is Offline")

    subscribed = false;
}

void publishGValueBatch(char *dataID) {
    protocolPublishData(dataID, gValueDataBatch);
}

/* endregion PROTOTYPE IMPLEMENTATIONS */
