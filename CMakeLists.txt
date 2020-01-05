cmake_minimum_required(VERSION 3.10)

project(TestTweetsStreaming)
add_executable( testproducer test_producer.cpp )
add_executable( testconsumer test_consumer.cpp )

find_package(CppKafka REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(testproducer CppKafka::cppkafka ${Boost_LIBRARIES})
target_link_libraries(testconsumer CppKafka::cppkafka ${Boost_LIBRARIES}) 

file(COPY Tweets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
