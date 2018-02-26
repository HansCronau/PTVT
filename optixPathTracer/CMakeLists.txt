#
# Path Traced Virtual Textures (PTVT)
# Copyright 2018 Hans Cronau
#
# File based on the Optix SDK optixPathTracer sample,
# Copyright 2016 Nvidia Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Cmake Options

# Scenes
option (SCENE_SPONZA "Uses Sponza as scene" ON)
option (SCENE_CORNELL_BOX "Uses Cornell Box as scene" OFF)

set(PT_DEFAULT_OUTPUT_WIDTH       "512"  CACHE STRING "The default (starting) width (in texels) of the output buffer.")
set(PT_DEFAULT_OUTPUT_HEIGHT      "512"  CACHE STRING "The default (starting) height (in texels) of the output buffer.")

set(VT_TILE_DETERMINATION_DEPTH   "4"    CACHE STRING "Max tile determination depth. (Does NOT determine memory allocation for differentials. See VT_DIFFERENTIAL_DEPTH instead.)")
set(VT_UNIQUE_TILID_BUFFER_LENGTH "2048" CACHE STRING "Length of the one dimensional unique tileID buffer. Should be more than (expected) max nr visible tiles per frame.")
set(VT_DIFFERENTIAL_DEPTH         "4"    CACHE STRING "Max path depth at which new differentials are created. More differentials linearly take up more memory.")

set(VT_MIN_TILE_MIPID             "0"    CACHE STRING "The lowest resolution mip level alsways to be loaded into the tile pool.")
set(VT_TILE_POOL_TEXELS_WIDE      "4096" CACHE STRING "The width (and height) of the tile pool texture (in texels).")
set(VT_KEEP_ALIVE_TIME            "4"    CACHE STRING "The number of frames to keep a previously visible tile in memory after becoming invisible.")
set(VT_MAX_TILE_UPLOADS_PER_FRAME "20"   CACHE STRING "The maximum number of tiles to be uploaded per frame.")

option (VT_TEST_ADDRESS_TRANSLATION "Testing - Sets program to test address translation code only." OFF)
option (VT_TEST_TILE_READER "Testing - Sets program to test tile reader code only." OFF)

option (VT_DEBUG_OPTIX_PRINT_ENABLED "Debug - Enables OptiX programs printing debugging info to console." OFF)
option (VT_DEBUG_PRINT_RENDER_LOOP "Debug - Prints vt render steps of every frame." OFF)
option (VT_DEBUG_PRINT_TILEIDS "Debug - Prints detected tileIDs every frame." OFF)
option (VT_DEBUG_TILE_BORDERS "Debug - Adding a red border around tiles is on by default." OFF)
option (VT_DEBUG_DIRECT_HIT_ONLY "Debug - Limits number of bounces to one per path per default." OFF)

option (VT_WRITE_TILE_COUNTS_TO_FILE "Write tile statistics to a log file every frame." OFF)
option (VT_WRITE_FRAME_TIMES_TO_FILE "Write frame times to a log file every frame." OFF)
option (VT_WRITE_FRAMES_TO_FILES "By default write framebuffer to file each frame." OFF)
set(VT_WRITE_FRAMES_TYPE ".png" CACHE STRING "The image type to store frames.")
set_property(
    CACHE VT_WRITE_FRAMES_TYPE
    PROPERTY STRINGS
    ".ppm" ".png" ".jpg"
)

set(VT_RESEARCH_METHOD "none" CACHE STRING "The research method to build.")
set_property(
    CACHE VT_RESEARCH_METHOD
    PROPERTY STRINGS
    "none"
    "speed - visible tiles"
    "speed - frame time - atlased"
    "speed - frame time - virtual"
    "speed - frame time - virtual without tile upload"
    "accuracy"
)

# Override configuration if research parameters are set. This is to prevent accidental misconfiguration.
if(NOT VT_RESEARCH_METHOD STREQUAL "none")

    # This is a research build.
    set(VT_RESEARCH_BUILD ON)

    # Enable Sponza and disable everything else.
    set(SCENE_SPONZA ON CACHE BOOL "Uses Sponza as scene" FORCE)
    set(SCENE_CORNELL_BOX OFF CACHE BOOL "Uses Cornell Box as scene" FORCE)
    set(VT_WRITE_TILE_COUNTS_TO_FILE OFF CACHE BOOL "Write tile statistics to a log file every frame." FORCE)
    set(VT_WRITE_FRAME_TIMES_TO_FILE OFF CACHE BOOL "Write frame times to a log file every frame." FORCE)
    set(VT_DEBUG_OPTIX_PRINT_ENABLED OFF CACHE BOOL "Debug - Enables OptiX programs printing debugging info to console." FORCE)
    set(VT_DEBUG_PRINT_RENDER_LOOP OFF CACHE BOOL "Debug - Prints vt render steps of every frame." FORCE)
    set(VT_DEBUG_PRINT_TILEIDS OFF CACHE BOOL "Debug - Prints detected tileIDs every frame." FORCE)
    set(VT_DEBUG_TILE_BORDERS OFF CACHE BOOL "Debug - Adding a red border around tiles is on by default." FORCE)
    set(VT_DEBUG_DIRECT_HIT_ONLY OFF CACHE BOOL "Debug - Limits number of bounces to one per path per default." FORCE)
    set(VT_TEST_ADDRESS_TRANSLATION OFF CACHE BOOL "Testing - Sets program to test address translation code only." FORCE)
    set(VT_TEST_TILE_READER OFF CACHE BOOL "Testing - Sets program to test tile reader code only." FORCE)
    set(VT_WRITE_FRAMES_TO_FILES OFF CACHE BOOL "By default write framebuffer to file each frame." FORCE)

    # Method-specific configuration.
    if     (VT_RESEARCH_METHOD STREQUAL "speed - visible tiles")
        set(VT_RESEARCH_METHOD_SPEED_VISIBLE_TILES ON)
        set(VT_WRITE_TILE_COUNTS_TO_FILE ON CACHE BOOL "Write tile statistics to a log file every frame." FORCE)

    elseif(VT_RESEARCH_METHOD STREQUAL "speed - frame time - atlased")
        set(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_ATLASED ON)
        set(VT_WRITE_FRAME_TIMES_TO_FILE ON CACHE BOOL "Write frame times to a log file every frame." FORCE)

    elseif(VT_RESEARCH_METHOD STREQUAL "speed - frame time - virtual")
        set(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL ON)
        set(VT_WRITE_FRAME_TIMES_TO_FILE ON CACHE BOOL "Write frame times to a log file every frame." FORCE)

    elseif(VT_RESEARCH_METHOD STREQUAL "speed - frame time - virtual without tile upload")
        set(VT_RESEARCH_METHOD_SPEED_FRAME_TIME_VIRTUAL_NO_TILE_UPLOAD ON)
        set(VT_WRITE_FRAME_TIMES_TO_FILE ON CACHE BOOL "Write frame times to a log file every frame." FORCE)

    elseif(VT_RESEARCH_METHOD STREQUAL "accuracy")
        set(VT_RESEARCH_METHOD_ACCURACY ON)
    endif()
    
    # Notify user of automated variable changes.
    message(WARNING "Research method \"${VT_RESEARCH_METHOD}\" was set. Note that some user options were overridden.")
endif()

# Configure sampleConfig.h with above configuration.
configure_file(ptvtConfig.h.in ptvtConfig.h)

# Required for new devil_cpp_wrapper and ILLoader.
set(DEVIL_LIBRARY_DIR "C:/cpplibs/DevIL Windows SDK/lib/x64/Release" CACHE PATH "Path to the libraries to link from DevIL.")
set(DEVIL_INCLUDE_DIR "C:/cpplibs/DevIL Windows SDK/include"         CACHE PATH "Path to the headers to include from DevIL.")

# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
if(GLUT_FOUND AND OPENGL_FOUND)

    include_directories(${GLUT_INCLUDE_DIR})
    add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

    OPTIX_add_sample_executable( optixPathTracer 
        optixPathTracer.cpp
        optixPathTracer.cu
        optixPathTracer.h

        # Added by Hans Cronau
        ${PROJECT_BINARY_DIR}/sampleConfig.h
        ${CMAKE_CURRENT_BINARY_DIR}/ptvtConfig.h
        DevIL/devil_cpp_wrapper.cpp
        DevIL/devil_cpp_wrapper.h
        ClosestHitManager.cpp
        ClosestHitManager.h
        vtAtlasXML.cpp
        vtAtlasXML.h
        vtHelpers.h
        vtManager.cpp
        vtManager.h
        vtMipIDTexture.cpp
        vtMipIDTexture.h
        vtPageTable.cpp
        vtPageTable.h
        vtTileDetermination.cu
        vtTileCache.cpp
        vtTileCache.h
        vtTileReader.cpp
        vtTileReader.h
        vtTilesXML.cpp
        vtTilesXML.h
        vtTriangleMesh.cu

        # These files are common among multiple samples
        parallelogram.cu
        random.h
        helpers.h # Required for (virtual texture) differentials
    )

    # Include own binary dir for configured ptvtConfig.h.
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    
    # Include DevIL headers and link DevIL libraries.
    include_directories(${DEVIL_INCLUDE_DIR})
    # DevIL.lib
    find_library(DEVIL_LIBRARY_PATH_DevIL DevIL ${DEVIL_LIBRARY_DIR})
    target_link_libraries( optixPathTracer ${DEVIL_LIBRARY_PATH_DevIL} )
    # ILU.lib
    find_library(DEVIL_LIBRARY_PATH_ILU ILU ${DEVIL_LIBRARY_DIR})
    target_link_libraries( optixPathTracer ${DEVIL_LIBRARY_PATH_ILU} )
    # ILUT.lib
    find_library(DEVIL_LIBRARY_PATH_ILUT ILUT ${DEVIL_LIBRARY_DIR})
    target_link_libraries( optixPathTracer ${DEVIL_LIBRARY_PATH_ILUT} )

    # Include and link PugiXML (3rd party)
    include_directories("${PROJECT_SOURCE_DIR}/pugixml")
    target_link_libraries ( optixPathTracer PugiXML )

else()
    # GLUT or OpenGL not found
    message("Disabling optixPathTracer, which requires GLUT and OpenGL.")
endif()
