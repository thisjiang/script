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
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "tests/test_template.h"

TEST(DO_GTEST, Case1) {
  ASSERT_TRUE(true);
  LOG(INFO) << "TEST Success\n";
}

TEST(DO_GTEST, test_template) {
  std::vector<bool> x = {true, false, true};
  test_func(x);
}

TEST(DO_GTEST, test_pow) { LOG(INFO) << "2^2=" << powf(2.0, 2.0); }
