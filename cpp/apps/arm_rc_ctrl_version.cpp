// Copyright (c) 2026 Hiroshi Atsuta
// SPDX-License-Identifier: GPL-3.0-only
//
// Minimal application slice: prints the library version and exits.

#include <iostream>

#include "arm_rc_ctrl/version.hpp"

int main() {
  std::cout << "arm_rc_ctrl " << arm_rc_ctrl::version() << '\n';
  return 0;
}
