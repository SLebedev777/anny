set(BINARY ${CMAKE_PROJECT_NAME}_tests)

set(SOURCES)

list(APPEND SOURCES
	"main.cpp"
	"VecTests.cpp"
	"MatrixTests.cpp"
	"DistanceTests.cpp"
	"HyperplaneTests.cpp"
	"GraphTests.cpp"
	"CsvLoaderTests.cpp"
	"FixedSizePriorityQueueTests.cpp"
	"VanillaKnnTests.cpp"
	"KDTreeTests.cpp"
	"AnnoyTests.cpp"
	"SkipListTests.cpp"
)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# VERY IMPORTANT!!! THIS IS NEEDED THAT GTEST WILL NOT OVERRIDE MAIN PROJECT'S COMPILING&LINKING SETTINGS!
if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(googletest)

enable_testing()

# Add source to this project's executable.
add_executable (${BINARY} ${SOURCES})

include(GoogleTest)
add_test(NAME ${BINARY} COMMAND ${BINARY})

target_include_directories(${BINARY} PUBLIC "../src")
target_link_libraries(${BINARY} gtest gtest_main)

