

add_custom_target(configure ALL)

function (_always_configure_file_ name source dest)
  add_custom_command(
    OUTPUT  "${dest}"
            ${extra_output}
    COMMAND "${CMAKE_COMMAND}"
            -E copy "${source}" "${dest}"
    MAIN_DEPENDENCY
            "${source}"
    DEPENDS "${source}"
    WORKING_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "copy ${name} file \"${source}\" -> \"${dest}\"")

  add_custom_target(configure-${name} ${all}
      DEPENDS "${dest}"
      SOURCES "${source}")
  add_dependencies(configure configure-${name})
endfunction ()

function (configure_file_always name source dest)
  set(extra_output
    "${dest}.noexist")
  _always_configure_file_(${name} "${source}" "${dest}" ${ARGN})
endfunction ()