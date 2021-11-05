import math


#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class LinearLr:
    def __init__(self, param_group, learning_rate: float, total_steps: int, delay: bool):
        self.total_steps = total_steps
        self.delay_steps = total_steps / 20 if delay else 0
        self.max_lr = learning_rate
        self.steps = 0
        self.param_group = param_group

    def __call__(self, _):
        self.steps += 1

        if self.steps < self.delay_steps:
            lr = 0.0
        elif self.steps < self.total_steps / 10:
            lr = self.max_lr * (self.steps - self.delay_steps) / (self.total_steps / 10 - self.delay_steps)
        else:
            max_lr = self.max_lr - self.max_lr / 100
            min_lr = self.max_lr / 100
            lr = max_lr * (math.cos(math.pi * (self.steps - self.total_steps / 10) / (self.total_steps * 9 / 10)) + 1) / 2 + min_lr
            #lr = self.max_lr * (self.total_steps - self.steps) / (self.total_steps * 9 / 10)

        # Safety first!
        if lr < 0.0:
            lr = 0.0

        self.param_group["lr"] = lr

    def lr(self) -> float:
        return self.param_group["lr"]
