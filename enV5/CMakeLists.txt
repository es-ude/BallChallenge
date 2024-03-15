cmake_minimum_required(VERSION 3.13)
include(extern/elastic-ai.runtime.enV5/enV5_init.cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

function(make_to_output_file target)
    # enable usb output, disable uart output
    pico_enable_stdio_usb(${target} 1)
    pico_enable_stdio_uart(${target} 0)
    # create map/bin/hex/uf2 file etc.
    pico_add_uf2_output(${target})
    # move u2f files after build to out directory
    file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ..)
    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/${relative_path}/${target}.uf2
            ${CMAKE_SOURCE_DIR}/out/${CMAKE_BUILD_TYPE}/${relative_path}/${target}.uf2)
endfunction()

if (DEBUG_OUTPUT)
    add_definitions(-DDEBUG_MODE)
endif ()

# add pico-sdk
include(extern/elastic-ai.runtime.enV5/extern/pico-sdk/pico_sdk_init.cmake)
# add FreeRTOS
add_subdirectory(extern/elastic-ai.runtime.enV5/extern/freeRTOS/Kernel/portable/ThirdParty/GCC/RP2040 FREERTOS_KERNEL)

project(BallChallenge C CXX ASM)

pico_sdk_init()
enV5_init()

if (TARGET tinyusb_device)
    add_subdirectory(src)
elseif (PICO_ON_DEVICE)
    message(WARNING "not building Project because TinyUSB submodule is not initialized in the SDK")
endif ()
