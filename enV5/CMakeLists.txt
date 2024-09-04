cmake_minimum_required(VERSION 3.21...3.24)
include(${CMAKE_CURRENT_LIST_DIR}/helper.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (DEBUG_OUTPUT)
    add_definitions(-DDEBUG_MODE)
endif ()

message(NOTICE "INFO: Building for enV5 hardware (Type: ${CMAKE_BUILD_TYPE})")

# include pico-sdk
set(PICO_SDK_FETCH_FROM_GIT on)
set(PICO_BOARD none)
set(PICO_PLATFORM rp2040)
set(PICOTOOL_FETCH_FROM_GIT_PATH ${CMAKE_SOURCE_DIR}/picotool)

include(pico_sdk_import.cmake)

project(BallChallenge C CXX ASM)

# load enV5 repo
load_enV5()

# initialize pico-sdk
pico_sdk_init()

if (TARGET tinyusb_device)
    # load libraries from runtime.enV5
    enV5_init()

    # add custom targets
    add_subdirectory(src)
elseif (PICO_ON_DEVICE)
    message(WARNING "not building Project because TinyUSB submodule is not initialized in the SDK")
endif ()
