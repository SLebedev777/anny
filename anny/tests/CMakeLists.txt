set(BINARY ${CMAKE_PROJECT_NAME}_tests)

set(SOURCES)

list(APPEND SOURCES
	"main.cpp"
	"VecTests.cpp"
	"MatrixTests.cpp"
	"DistanceTests.cpp"
)

# VERY IMPORTANT!!! THIS IS NEEDED THAT GTEST WILL NOT OVERRIDE MAIN PROJECT'S COMPILING&LINKING SETTINGS!
if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# Add source to this project's executable.
add_executable (${BINARY} ${SOURCES})

add_test(NAME ${BINARY} COMMAND ${BINARY})

target_include_directories(${BINARY} PUBLIC "../src")
target_link_libraries(${BINARY} gtest gtest_main)

