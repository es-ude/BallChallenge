## region Data Collection
# Target provides Data Collection Service for training
add_executable(data_collection DataCollection.c)
target_link_libraries(data_collection
        pico_stdlib
        freeRtosUtils
        common_lib
        enV5_hwc
        network_lib
        espBroker_lib
        sensor_lib_adxl345b)
create_enV5_executable(data_collection)
## endregion Data Collection
##################################################
## region Wrist
# Target provides FPGA accelerated predictions
add_executable(nn_test NeuralNetworkTest.c)
target_link_libraries(nn_test
        pico_stdlib
        freeRtosUtils
        common_lib
        enV5_hwc
        network_lib
        espBroker_lib
        http_lib
        flash_lib
        fpga_configuration_handler
        stub_v2_lib)
create_enV5_executable(nn_test)
## endregion Wrist
