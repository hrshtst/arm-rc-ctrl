// Copyright (c) 2026 Hiroshi Atsuta
// SPDX-License-Identifier: GPL-3.0-only

#include <catch2/catch_test_macros.hpp>

#include <cctype>
#include <string>

#include "arm_rc_ctrl/version.hpp"

TEST_CASE("version is a non-empty MAJOR.MINOR.PATCH string", "[version]") {
  const std::string v{arm_rc_ctrl::version()};
  REQUIRE_FALSE(v.empty());

  int dots = 0;
  for (const char c : v) {
    if (c == '.') {
      ++dots;
    } else {
      REQUIRE(std::isdigit(static_cast<unsigned char>(c)) != 0);
    }
  }
  REQUIRE(dots == 2);
}
