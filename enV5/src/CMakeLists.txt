## Wrist
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
make_to_output_file(wrist)

## Data Collection
add_executable(data_collection DataCollection.c)
target_link_libraries(data_collection
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
make_to_output_file(data_collection)

