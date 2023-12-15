#define SOURCE_FILE "MAIN"

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
#include "NetworkConfiguration.h"
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
#include <string.h>

/* region VARIABLES/DEFINES */

/* region FLASH */

spi_t spiToFlash = {.spi = spi0, .baudrate = 5000000, .sckPin = 2, .mosiPin = 3, .misoPin = 0};
uint8_t flashChipSelectPin = 1;

/* endregion FLASH */

/* region MQTT */

#define G_VALUE_BATCH_SECONDS 3

bool newBatchAvailable = false;
bool newBatchRequested = false;
char *gValueDataBatch;
char *twinID;

/* endregion MQTT */

/* endregion VARIABLES/DEFINES */

/* region PROTOTYPES */

/* region HARDWARE */

/// initialize hardware (watchdog, UART, ESP, Wi-Fi, MQTT, I²C, sensors, ...)
void init(void);

/* endregion HARDWARE */

/* region FreeRTOS TASKS */

void setupTask(void);

_Noreturn void publishValueBatchesTask(void);

_Noreturn void getGValueTask(void);

/* endregion FreeRTOS TASKS */

/* region MQTT */

void publishGValueBatch(char *dataID);

void receiveCsvRequest(__attribute__((unused)) posting_t posting);

/* endregion MQTT */

/* endregion HEADER */

/* region PROTOTYPE IMPLEMENTATIONS */

int main() {
    init();
    env5HwLedsAllOn();

    freeRtosTaskWrapperRegisterTask(setupTask, "setupMqtt", 1, FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(publishValueBatchesTask, "publishValueBatchesTask", 1,
                                    FREERTOS_CORE_0);
    freeRtosTaskWrapperRegisterTask(getGValueTask, "getGValueTask", 1, FREERTOS_CORE_1);
    freeRtosTaskWrapperStartScheduler();
}

void init(void) {
    env5HwLedsInit();

    // check if we crash last time -> reboot into boot rom mode
    if (watchdog_enable_caused_reboot()) {
        reset_usb_boot(0, 0);
    }

    // init IO
    stdio_init_all();
    // waits for usb connection, REMOVE to continue without waiting for connection
    // while ((!stdio_usb_connected())) {}
    // initialize ESP over UART
    espInit();

    // initialize Wi-Fi and MQTT broker
    networkTryToConnectToNetworkUntilSuccessful(networkCredentials);
    mqttBrokerConnectToBrokerUntilSuccessful(mqttHost, "eip://uni-due.de/es", "enV5");

    adxl345bErrorCode_t errorADXL = adxl345bInit(i2c1, ADXL345B_I2C_ALTERNATE_ADDRESS);
    i2c_set_baudrate(i2c1, 2000000);
    if (errorADXL == ADXL345B_NO_ERROR)
        PRINT_DEBUG("Initialised ADXL345B.")
    else
        PRINT("Initialise ADXL345B failed; adxl345b_ERROR: %02X", errorADXL)

    // initialize FPGA and flash
    flashInit(&spiToFlash, flashChipSelectPin);
    env5HwInit();
    fpgaConfigurationHandlerInitialize();

    // create FreeRTOS task queue
    freeRtosQueueWrapperCreate();

    // enables watchdog timer (5s)
    // watchdog_enable(5000, 1);
}


void setupTask(void) {
    publishAliveStatusMessage("g-value,time");
    protocolSubscribeForCommand("MEASUREMENTS", (subscriber_t){.deliver = receiveCsvRequest});
    return;
}

_Noreturn void getGValueTask(void) {
    PRINT_DEBUG("START GVALUE TASK")

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
            freeRtosTaskWrapperTaskSleep(100);
            continue;
        }
        newBatchRequested = false;

        env5HwLedsInit();
        protocolPublishData("time", "3");
        gpio_put(GPIO_LED0, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        protocolPublishData("time", "2");
        gpio_put(GPIO_LED1, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        protocolPublishData("time", "1");
        gpio_put(GPIO_LED2, 1);
        freeRtosTaskWrapperTaskSleep(1000);
        env5HwLedsAllOff();
        freeRtosTaskWrapperTaskSleep(250);
        protocolPublishData("time", "0");
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
            if (count >= batchSize * G_VALUE_BATCH_SECONDS)
                continue;
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

_Noreturn void publishValueBatchesTask(void) {
    PRINT_DEBUG("START BATCH TASK")

    while (true) {
        freeRtosTaskWrapperTaskSleep(100);
        if (newBatchAvailable) {
            publishGValueBatch("g-value");
            newBatchAvailable = false;
            env5HwLedsAllOn();
            PRINT_DEBUG("Published G-Values (sec: %lu)", lastPublished)
        }
    }
}

void receiveCsvRequest(__attribute__((unused)) posting_t posting) {
    newBatchRequested = true;
    PRINT_DEBUG("Batch requested!")
}

void publishGValueBatch(char *dataID) {
    protocolPublishData(dataID, gValueDataBatch);
}

/* endregion PROTOTYPE IMPLEMENTATIONS */
