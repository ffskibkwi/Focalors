file(GLOB CASE_SOURCES "*.cpp")

foreach(src ${CASE_SOURCES})
  get_filename_component(exec_name ${src} NAME_WE)
  add_executable(${exec_name} ${src})
  target_link_libraries(${exec_name} PRIVATE Solver)
endforeach()


