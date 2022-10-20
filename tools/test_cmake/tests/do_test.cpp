// Copyright (c) 2022 thisjiang Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <cmath>
#include <optional>
#include <string>

int main() {
  // test C++17
  std::optional<std::string> ss;
  ss = "C++17 supported";
  if (ss) {
    LOG(INFO) << ss.value();
  } else {
    LOG(INFO) << "C++17 not supported\n";
  }

  LOG(INFO) << (0 >= 0UL && 0 < 800UL);

  return 0;
}
