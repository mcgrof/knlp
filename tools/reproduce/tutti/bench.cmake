# SPDX-License-Identifier: MIT
# Included through CMAKE_PROJECT_Tutti_INCLUDE at Tutti's root project().
# Targets defined later by Tutti supply their transitive API/CUDA requirements.
add_executable(knlp_tutti_io "${CMAKE_CURRENT_LIST_DIR}/transfer.cpp")
target_link_libraries(knlp_tutti_io PRIVATE tutti_runtime tutti_api)
target_compile_features(knlp_tutti_io PRIVATE cxx_std_17)
target_compile_options(knlp_tutti_io PRIVATE -Wall -Wextra)
set_target_properties(knlp_tutti_io PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
