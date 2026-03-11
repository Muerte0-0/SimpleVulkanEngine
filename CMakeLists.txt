# CMakeList.txt : CMake project for SimpleVulkanEngine, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.29)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

# Add option to enable/disable C++ 20 module
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" ON)

# Enable C++ module dependency scanning only if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

find_package (glfw3 REQUIRED)
find_package (glm REQUIRED)
find_package (Vulkan 1.4.335 REQUIRED)
find_package (tinyobjloader REQUIRED)

set(tinyobjloader_INCLUDEDIR ${tinyobjloader_INCLUDE_DIRS})

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

project (SimpleVulkanEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

function(add_shader_compile_target TARGET)

    set(SHADER_DIR ${CMAKE_SOURCE_DIR}/assets/shaders)
    set(COMPILE_SCRIPT ${SHADER_DIR}/Compile.bat)

    add_custom_target(${TARGET}
        COMMAND ${CMAKE_COMMAND} -E echo "Compiling shaders..."
        COMMAND cmd /c ${COMPILE_SCRIPT}
        WORKING_DIRECTORY ${SHADER_DIR}
        COMMENT "Running shader compile script"
        VERBATIM
    )

endfunction()

file(GLOB_RECURSE SOURCES 
${PROJECT_SOURCE_DIR}/src/*.cpp
Vulkan::cppm
glm::glm
tinyobjloader::tinyobjloader
)

add_executable (${PROJECT_NAME} ${SOURCES})

add_shader_compile_target(Shaders)
add_dependencies(${PROJECT_NAME} Shaders)

target_include_directories(${PROJECT_NAME} PUBLIC
${PROJECT_SOURCE_DIR}
${Vulkan_INCLUDE_DIR}
${GLFW3_INCLUDE_DIR}
${glm_INCLUDE_DIR}
${STB_INCLUDEDIR}
${tinyobjloader_INCLUDEDIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
${VULKAN_LIBRARIES}
${GLFW3_LIBRARY}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE 
NOMINMAX
)