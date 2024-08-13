/*!
 * This implementation intends to provide an implementation to test the prediction on the FPGA
 * It is intended to be used with the CSV Service as a backend (See README.adoc).
 */

#define SOURCE_FILE "INFERENCE-APP"

// external headers
#include <string.h>

// pico-sdk headers
#include "FpgaConfigurationHandler.h"
#include "hardware/spi.h"
#include "hardware/watchdog.h"
#include "pico/bootrom.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"

// internal headers
#include "Common.h"
#include "EnV5HwConfiguration.h"
#include "EnV5HwController.h"
#include "Esp.h"
#include "FreeRtosMutexWrapper.h"
#include "FreeRtosQueueWrapper.h"
#include "FreeRtosTaskWrapper.h"
#include "MqttBroker.h"
#include "Network.h"
#include "Posting.h"
#include "Protocol.h"
#include "stub.h"

/* region VARIABLES/DEFINES */

#define EIP_BASE "eip://uni-due.de/es"
#define EIP_DEVICE_ID "predict01"
status_t status = {
    .id = EIP_DEVICE_ID, .type = "enV5", .state = STATUS_STATE_ONLINE, .data = "prediction"};

typedef enum {
    DATA_VALUE,
    COMMAND_RESPONSE,
} pubType_t;
typedef struct publishRequest {
    pubType_t pubType;
    char *topic;
    char *data;
} publishRequest_t;

typedef struct downloadRequest {
    char *url;            //!< URL for download via HTTPGet
    size_t binFileSize;   //!< Size of a BIN file in Bytes
    uint32_t startSector; //!< Flash Sector ID (Starting with 1)
    uint8_t *hash;        //!< pointer to SHA256 Hash for validation
} downloadRequest_t;

typedef struct testRequest {
    size_t numberOfInputs;
    float *data;   //!< input data to evaluate neural network
    uint8_t *hash; //!< hash to identify neural network
} testRequest_t;

queue_t receivedPosts;
queue_t publishRequests;
queue_t downloadRequests;
queue_t testRequests;

mutex_t espOccupied;

spiConfiguration_t flashSpi = {
    .spiInstance = FLASH_SPI_MODULE,
    .baudrate = FLASH_SPI_BAUDRATE,
    .csPin = FLASH_SPI_CS,
    .sckPin = FLASH_SPI_CLOCK,
    .misoPin = FLASH_SPI_MISO,
    .mosiPin = FLASH_SPI_MOSI,
};
flashConfiguration_t flash = {
    .flashSpiConfiguration = &flashSpi,
    .flashBytesPerSector = FLASH_BYTES_PER_SECTOR,
    .flashBytesPerPage = FLASH_BYTES_PER_PAGE,
};

/* endregion VARIABLES/DEFINES */

/* region HELPER FUNCTIONS */

void initialize(void) {
    // check if we crash last time → if true, reboot into boot rom mode
    if (watchdog_enable_caused_reboot()) {
        reset_usb_boot(0, 0);
    }

    env5HwControllerInit();

    // initialize I/O for Debug purposes
#ifndef NDEBUG
    stdio_init_all();
    while ((!stdio_usb_connected())) { /* wait for serial connection */
    }
#endif

    espInit(); // initialize Wi-Fi chip
    networkTryToConnectToNetworkUntilSuccessful();
    mqttBrokerConnectToBrokerUntilSuccessful(EIP_BASE, EIP_DEVICE_ID);
}

_Noreturn void watchdogTask(void) {
    watchdog_enable(10000, 1); // enables watchdog timer (10 seconds)

    while (1) {
        watchdog_update();                  // watchdog update needs to be performed frequent
        freeRtosTaskWrapperTaskSleep(1000); // sleep for 1 second
    }
}

void deliver(posting_t posting) {
    freeRtosQueueWrapperPushFromInterrupt(receivedPosts, &posting);
}
downloadRequest_t parseDownloadRequest(char *request) {
    PRINT("RECEIVED FLASH REQUEST");

    /* region parse length */
    char *sizeStart = strstr(request, "SIZE:") + 5;
    size_t length = strtol(sizeStart, NULL, 10);
    /* endregion parse length */
    /* region parse sector-ID */
    char *positionStart = strstr(request, "POSITION:") + 9;
    uint32_t position = strtol(positionStart, NULL, 10);
    /* endregion parse sector-ID */
    /* region parse url */
    char *urlStart = strstr(request, "URL:") + 4;
    char *urlRaw = strtok(urlStart, ";");
    size_t urlLength = strlen(urlRaw);
    char *url = malloc(urlLength + 1);
    strcpy(url, urlRaw);
    /* endregion parse url */

    PRINT_DEBUG("URL: %s", url);
    PRINT_DEBUG("LENGTH: %zu", length);
    PRINT_DEBUG("SECTOR 0: %lu", position);

    return (downloadRequest_t){
        .url = url, .startSector = position, .binFileSize = length, .hash = NULL};
}
testRequest_t parseTestRequest(char *request) {
    PRINT("Received Test Request");

    /* region parse number of input values */
    char *inputCountStart = strstr(request, "LENGTH:") + 6;
    size_t inputCount = strtol(inputCountStart, NULL, 10);
    /* endregion parse number of input values */
    /* region parse data */
    float *inputs = malloc(inputCount * sizeof(float));
    float *next = inputs;
    char *rawData = strtok(strstr(request, "DATA:") + 5, ";");
    char *currentRawValue = strtok(rawData, ",");
    while (currentRawValue != NULL) {
        *next = strtof(currentRawValue, NULL);
        next += 1;
        currentRawValue = strtok(NULL, ",");
    }
    /* endregion parse data */

    return (testRequest_t){.numberOfInputs = inputCount, .data = inputs, .hash = NULL};
}
_Noreturn void handleReceivedPostingsTask(void) {
    freeRtosTaskWrapperTaskSleep(500);
    protocolSubscribeForCommand("FLASH", (subscriber_t){.deliver = deliver});
    protocolSubscribeForCommand("TEST", (subscriber_t){.deliver = deliver});

    while (1) {
        posting_t post;
        if (freeRtosQueueWrapperPop(receivedPosts, &post)) {
            PRINT("Received Message: '%s' via topic '%s'", post.data, post.topic);
            if (NULL != strstr(post.topic, "/DO/TEST")) {
                downloadRequest_t request = parseDownloadRequest(post.data);
                freeRtosQueueWrapperPush(downloadRequests, &request);
            } else if (NULL != strstr(post.topic, "/DO/FLASH")) {
                testRequest_t request = parseTestRequest(post.data);
                freeRtosQueueWrapperPush(testRequests, &request);
            }

            free(post.topic);
            free(post.data);
        }
        freeRtosTaskWrapperTaskSleep(500);
    }
}

_Noreturn void handlePublishTask(void) {
    publishAliveStatusMessageWithMandatoryAttributes(status);

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
            case COMMAND_RESPONSE:
                freeRtosMutexWrapperLock(espOccupied);
                protocolPublishCommandResponse(request.topic, strcmp(request.data, "FAILED"));
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

void downloadBinary(downloadRequest_t *request) {
    freeRtosMutexWrapperLock(espOccupied);
    fpgaConfigurationHandlerError_t error = fpgaConfigurationHandlerDownloadConfigurationViaHttp(
        &flash, request->url, request->binFileSize, request->startSector);
    freeRtosMutexWrapperUnlock(espOccupied);
    free(request->url);

    publishRequest_t downloadDonePosting;
    downloadDonePosting.pubType = COMMAND_RESPONSE;
    downloadDonePosting.topic = calloc(6, sizeof(char));
    strcpy(downloadDonePosting.topic, "FLASH");
    if (FPGA_RECONFIG_NO_ERROR == error) {
        downloadDonePosting.data = calloc(8, sizeof(char));
        strcpy(downloadDonePosting.data, "SUCCESS");
        freeRtosQueueWrapperPush(publishRequests, &downloadDonePosting);
    }
    downloadDonePosting.data = calloc(7, sizeof(char));
    strcpy(downloadDonePosting.data, "FAILED");
    freeRtosQueueWrapperPush(publishRequests, &downloadDonePosting);
}
void executeTest(testRequest_t *request) {
    env5HwControllerFpgaPowersOn();

    int8_t *prediction = malloc(2 * sizeof(int8_t));
    modelPredict((int8_t *)request->data, (request->numberOfInputs) * sizeof(float), prediction, 2);
    free(request->data);

    publishRequest_t testDone = {.pubType = COMMAND_RESPONSE, .topic = "TEST", .data = "SUCCESS"};
    freeRtosQueueWrapperPush(publishRequests, &testDone);

    publishRequest_t testResult = {
        .pubType = DATA_VALUE, .topic = "prediction", .data = prediction};
    freeRtosQueueWrapperPush(publishRequests, &testResult);
}
_Noreturn void handleFpgaTask(void) {
    while (1) {
        downloadRequest_t downloadRequest;
        if (freeRtosQueueWrapperPop(downloadRequests, &downloadRequest)) {
            downloadBinary(&downloadRequest);
            freeRtosTaskWrapperTaskSleep(500);
        }

        testRequest_t testRequest;
        if (freeRtosQueueWrapperPop(testRequests, &testRequest)) {
            executeTest(&testRequest);
        }

        freeRtosTaskWrapperTaskSleep(1000);
    }
}

/* endregion HELPER FUNCTIONS */

int main() {
    initialize();

    env5HwControllerLedsAllOn();

    receivedPosts = freeRtosQueueWrapperCreate(10, sizeof(posting_t));
    publishRequests = freeRtosQueueWrapperCreate(10, sizeof(publishRequest_t));
    downloadRequests = freeRtosQueueWrapperCreate(2, sizeof(downloadRequest_t));
    testRequests = freeRtosQueueWrapperCreate(2, sizeof(testRequest_t));

    espOccupied = freeRtosMutexWrapperCreate();

    freeRtosTaskWrapperRegisterTask(watchdogTask, "watchdog", configMAX_PRIORITIES / 2,
                                    FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(handleReceivedPostingsTask, "receiver",
                                    configMAX_PRIORITIES / 2, FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(handlePublishTask, "sender", configMAX_PRIORITIES,
                                    FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(handleFpgaTask, "fpga_controller", configMAX_PRIORITIES,
                                    FREERTOS_CORE_1);

    freeRtosTaskWrapperStartScheduler();
}
