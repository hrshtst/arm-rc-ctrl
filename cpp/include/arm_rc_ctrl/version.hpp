// Copyright (c) 2026 Hiroshi Atsuta
// SPDX-License-Identifier: GPL-3.0-only

#ifndef ARM_RC_CTRL_VERSION_HPP
#define ARM_RC_CTRL_VERSION_HPP

#include <string_view>

namespace arm_rc_ctrl {

/// Semantic version of the C++ library ("MAJOR.MINOR.PATCH"), set by CMake.
[[nodiscard]] std::string_view version() noexcept;

}  // namespace arm_rc_ctrl

#endif  // ARM_RC_CTRL_VERSION_HPP
