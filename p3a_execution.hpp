#pragma once

#include "p3a_macros.hpp"

namespace p3a {

class serial_execution {
};

inline constexpr serial_execution serial = {};

class local_execution {
};

inline constexpr local_execution local = {};

}
