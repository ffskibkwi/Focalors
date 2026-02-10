function(add_cases_recursive)
  # Recursively find all .cpp files relative to the current source directory
  file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
  
  foreach(SOURCE_REL_PATH IN LISTS SOURCES)
    # Construct the full path to the source file
    set(SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_REL_PATH}")
    
    # Split the relative path into directory and filename components
    get_filename_component(SOURCE_DIR "${SOURCE_REL_PATH}" DIRECTORY)
    get_filename_component(SOURCE_NAME_WE "${SOURCE_REL_PATH}" NAME_WE)  # Removes .cpp extension
    
    # Generate a unique target name by replacing path separators with underscores
    # Example: tests/math/example.cpp -> tests_math_example (without _cpp suffix)
    if(SOURCE_DIR STREQUAL "")
      set(TARGET_NAME "${SOURCE_NAME_WE}")
    else()
      string(REPLACE "/" "_" DIR_NAME "${SOURCE_DIR}")
      set(TARGET_NAME "${DIR_NAME}_${SOURCE_NAME_WE}")
    endif()
    
    # Create an executable target from the single source file
    add_executable(${TARGET_NAME} ${SOURCE_PATH})
    target_link_libraries(${TARGET_NAME} PRIVATE Solver)
  endforeach()
endfunction()

# Invoke the function to process current directory and all subdirectories
add_cases_recursive()