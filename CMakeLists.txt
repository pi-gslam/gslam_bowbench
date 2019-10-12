cmake_minimum_required(VERSION 2.8)

include(PICMake)

include_directories(DBoW2/include/DBoW2 src/include)
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native")


OPTION(USE_AVX "Set on/off" ON)
OPTION(USE_SSE3 "Set on/off" ON)

if(USE_O2)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    endif()
    if(USE_FAST_MATH)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -ffast-math")
    endif()
    if(USE_POWERPC)
       set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()
    if(USE_MMX)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mmmx")
       add_definitions(-DUSE_MMX)
    endif()
    if(USE_SSE)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse ")
       add_definitions(-DUSE_SSE)
    endif()
    if(USE_SSE2)
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse2")
       add_definitions(-DUSE_SSE2)
    endif()
    if(USE_SSE3 AND NOT MINGW) # SSE3 should be disabled under MingW because it generates compiler errors
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse3")
       add_definitions(-DUSE_SSE3)
    endif()

    if(USE_SSE4 AND NOT MINGW) # SSE3 should be disabled under MingW because it generates compiler errors
       set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse4.1")
       add_definitions(-DUSE_SSE4)
    endif()

    IF(USE_AVX) 
      set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mavx")
     add_definitions(-DUSE_AVX)
    endif()

pi_add_target(gslam_bowbench SHARED DBow3/src DBoW2/src fbow/src src REQUIRED GSLAM GLEW GLUT OpenGL OpenCV System DEPENDENCY X11)
pi_report_target()
