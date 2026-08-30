// Copyright (c) 2026 Hiroshi Atsuta
// SPDX-License-Identifier: GPL-3.0-only

#include "arm_rc_ctrl/version.hpp"

#ifndef ARM_RC_CTRL_VERSION
#error "ARM_RC_CTRL_VERSION must be defined by the build system"
#endif

namespace arm_rc_ctrl {

std::string_view version() noexcept { return ARM_RC_CTRL_VERSION; }

}  // namespace arm_rc_ctrl
