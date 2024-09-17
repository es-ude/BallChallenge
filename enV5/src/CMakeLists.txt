##################################################
## region Data Collection BMI323
# Target provides Data Collection Service for training
add_executable(data_collection_bmi323
        ${CMAKE_CURRENT_LIST_DIR}/DataCollectionBmi323.c)
target_link_libraries(data_collection_bmi323
        pico_stdlib
        hardware_spi

        CException

        common_lib
        enV5_hw_controller
        enV5_hw_configuration
        freeRtosUtils
        network_lib
        espBroker_lib
        sensor_lib_bmi323
)
target_compile_definitions(data_collection_bmi323 PRIVATE PICO_DEBUG_MALLOC=1)
create_enV5_executable(data_collection_bmi323)
## endregion Data Collection BMI323
##################################################
## region Data Collection
# Target provides Data Collection Service for training
add_executable(data_collection_adxl345b DataCollectionAdxl345b.c)
target_link_libraries(data_collection_adxl345b
        pico_stdlib
        freeRtosUtils
        common_lib
        enV5_hw_controller
        enV5_hw_configuration
        network_lib
        espBroker_lib
        sensor_lib_adxl345b)
create_enV5_executable(data_collection_adxl345b)
## endregion Data Collection ADXL345B
##################################################
## region Wrist
# Target provides FPGA accelerated predictions
add_executable(nn_test NeuralNetworkTest.c)
target_link_libraries(nn_test
        pico_stdlib
        freeRtosUtils
        common_lib
        enV5_hw_configuration
        enV5_hw_controller
        network_lib
        espBroker_lib
        http_lib
        flash_lib
        fpga_configuration_handler
        stub_v2_lib)
create_enV5_executable(nn_test)
## endregion Wrist
##################################################