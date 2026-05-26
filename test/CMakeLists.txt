cmake_minimum_required(VERSION 3.10)
enable_testing()

set_source_files_properties(context.cpp  device.cpp  event.cpp  graph.cpp  launch.cpp  memory.cpp  module.cpp  stream.cpp PROPERTIES LANGUAGE CXX)

add_executable(stream stream.cpp)
add_executable(device device.cpp)
add_executable(context context.cpp)
add_executable(event event.cpp)
add_executable(graph graph.cpp)
add_executable(launch launch.cpp)
add_executable(memory memory.cpp)
add_executable(module module.cpp)

set(TEST_TARGETS stream device context event graph launch memory module)

foreach(t ${TEST_TARGETS})
    target_link_libraries(${t} pthread UPTKrt UPTKdrt)
endforeach()

add_test(NAME UPTKstreamCreate COMMAND stream)
add_test(NAME UPTKdeviceCreate COMMAND device)
add_test(NAME UPTKcontextCreate COMMAND context)
add_test(NAME UPTKeventCreate COMMAND event)
add_test(NAME UPTKgraphCreate COMMAND graph)
add_test(NAME UPTKlaunchCreate COMMAND launch)
add_test(NAME UPTKmemoryCreate COMMAND memory)
add_test(NAME UPTKmoduleCreate COMMAND module)
