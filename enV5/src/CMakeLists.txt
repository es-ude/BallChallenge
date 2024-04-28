## region Data Collection
# Target provides Data Collection Service for training
add_executable(data_collection DataCollection.c)
target_link_libraries(data_collection
        pico_stdlib
        hardware_i2c
        hardware_watchdog
        pico_bootrom
        common_lib
        enV5_hwc
        freeRtosUtils
        esp_lib
        network_lib
        espBroker_lib
        sensor_lib_adxl345b)
create_enV5_executable(data_collection)
## endregion Data Collection
##################################################
## region Wrist
# Target provides FPGA accelerated predictions
add_executable(wrist Wrist.c)
target_link_libraries(wrist
        pico_stdlib
        http_lib
        freeRtosUtils
        network_lib
        espBroker_lib
        fpga_configuration_handler
        flash_lib
        middleware_lib
        spi_lib
        sensor_lib_adxl345b
        enV5_hwc)
create_enV5_executable(wrist)
## endregion Wrist
