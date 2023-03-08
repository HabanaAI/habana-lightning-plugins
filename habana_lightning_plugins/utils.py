# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import habana_frameworks.torch.hpu as torch_hpu

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def get_device_type():
    return htexp._get_device_type()

def is_gaudi():
    return torch_hpu.get_device_name() == "GAUDI"

def is_gaudi2():
    return torch_hpu.get_device_name() == "GAUDI2"

def get_device_string():
    if is_gaudi():
        return "gaudi"
    elif is_gaudi2():
        return "gaudi2"
    else:
        raise ValueError("Unsupported device")