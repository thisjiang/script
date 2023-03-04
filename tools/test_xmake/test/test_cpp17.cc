// Copyright (c) 202 thisjiang Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <string>

TEST(HELLO, CPP17) {
  // test C++17
  std::optional<std::string> ss;
  ss = "C++17 supported";
  if (ss) {
    std::cout << ss.value() << std::endl;
  } else {
    std::cout << "C++17 not supported\n" << std::endl;
  }

  std::cout << (0 >= 0UL && 0 < 800UL) << std::endl;
}
