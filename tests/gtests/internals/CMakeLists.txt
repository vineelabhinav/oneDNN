#===============================================================================
# Copyright 2020-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

set(TEST_EXE test_internals)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "/fp:precise")
    append_if(UNIX  CMAKE_CXX_FLAGS "-fp-model precise")
endif()

file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
list(APPEND TEST_SOURCES ${MAIN_SRC_GTEST})

# Remove X64-specific tests
if(NOT DNNL_TARGET_ARCH STREQUAL "X64" OR DNNL_CPU_RUNTIME STREQUAL "NONE")
    list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_brgemm.cpp)
    list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_float8.cpp)
endif()

if(DNNL_ENABLE_MAX_CPU_ISA)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_MAX_CPU_ISA)
endif()

if(DNNL_ENABLE_CPU_ISA_HINTS)
    add_definitions_with_host_compiler(-DDNNL_ENABLE_CPU_ISA_HINTS)
endif()

# Register separate test targets to preserve testing environment and allow
# desired functionality to be tested properly since env vars are read only once
# per binary run.
register_exe(${TEST_EXE}_env_vars_dnnl
        "${MAIN_SRC_GTEST};${CMAKE_CURRENT_SOURCE_DIR}/test_env_vars_dnnl.cpp"
        "test" "dnnl_gtest")
list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_env_vars_dnnl.cpp)
register_exe(${TEST_EXE}_env_vars_onednn
        "${MAIN_SRC_GTEST};${CMAKE_CURRENT_SOURCE_DIR}/test_env_vars_onednn.cpp"
        "test" "dnnl_gtest")
list(REMOVE_ITEM TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_env_vars_onednn.cpp)

register_exe(${TEST_EXE} "${TEST_SOURCES}" "test" "dnnl_gtest")
