include(Catch)

add_executable(arm_rc_ctrl_tests version_test.cpp)
target_link_libraries(arm_rc_ctrl_tests PRIVATE arm_rc_ctrl::arm_rc_ctrl arm_rc_ctrl_warnings Catch2::Catch2WithMain)
catch_discover_tests(arm_rc_ctrl_tests)

# The application must run headless and report the library version.
add_test(NAME arm_rc_ctrl_version_app COMMAND arm_rc_ctrl_version)
set_tests_properties(arm_rc_ctrl_version_app PROPERTIES
  PASS_REGULAR_EXPRESSION "^arm_rc_ctrl ${PROJECT_VERSION}\n$")
