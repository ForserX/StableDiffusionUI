# Copyright 2020 The Nod Team. All rights reserved.
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

# All the iree_vulkan related functionalities go here.

from os import linesep
from repo.shark.iree_utils._common import run_cmd
import iree.runtime as ireert
from sys import platform
from repo.shark.iree_utils.vulkan_target_env_utils import get_vulkan_target_env_flag


def get_vulkan_device_name():
    vulkaninfo_dump = run_cmd("vulkaninfo").split(linesep)
    vulkaninfo_list = [s.strip() for s in vulkaninfo_dump if "deviceName" in s]
    if len(vulkaninfo_list) == 0:
        raise ValueError("No device name found in VulkanInfo!")
    if len(vulkaninfo_list) > 1:
        print("Following devices found:")
        for i, dname in enumerate(vulkaninfo_list):
            print(f"{i}. {dname}")
        print(f"Choosing first one: {vulkaninfo_list[0]}")
    return vulkaninfo_list[0]


def get_os_name():
    if platform.startswith("linux"):
        return "linux"
    elif platform == "darwin":
        return "macos"
    elif platform == "win32":
        return "windows"
    else:
        print("Cannot detect OS type, defaulting to linux.")
        return "linux"


def get_vulkan_target_triple(device_name):
    """This method provides a target triple str for specified vulkan device.

    Args:
        device_name (str): name of the hardware device to be used with vulkan

    Returns:
        str or None: target triple or None if no match found for given name
    """
    system_os = get_os_name()
    # Apple Targets
    if all(x in device_name for x in ("Apple", "M1")):
        triple = "m1-moltenvk-macos"
    elif all(x in device_name for x in ("Apple", "M2")):
        triple = "m1-moltenvk-macos"

    # Nvidia Targets
    elif all(x in device_name for x in ("RTX", "2080")):
        triple = f"turing-rtx2080-{system_os}"
    elif all(x in device_name for x in ("A100", "SXM4")):
        triple = f"ampere-a100-{system_os}"
    elif all(x in device_name for x in ("RTX", "3090")):
        triple = f"ampere-rtx3090-{system_os}"
    elif all(x in device_name for x in ("RTX", "3080")):
        triple = f"ampere-rtx3080-{system_os}"
    elif all(x in device_name for x in ("RTX", "3070")):
        triple = f"ampere-rtx3070-{system_os}"
    elif all(x in device_name for x in ("RTX", "3060")):
        triple = f"ampere-rtx3060-{system_os}"
    elif all(x in device_name for x in ("RTX", "3050")):
        triple = f"ampere-rtx3050-{system_os}"
    # We use ampere until lovelace target triples are plumbed in.
    elif all(x in device_name for x in ("RTX", "4090")):
        triple = f"ampere-rtx4090-{system_os}"
    elif all(x in device_name for x in ("RTX", "4080")):
        triple = f"ampere-rtx4080-{system_os}"
    elif all(x in device_name for x in ("RTX", "4070")):
        triple = f"ampere-rtx4070-{system_os}"
    elif all(x in device_name for x in ("RTX", "4000")):
        triple = f"turing-rtx4000-{system_os}"
    elif all(x in device_name for x in ("RTX", "5000")):
        triple = f"turing-rtx5000-{system_os}"
    elif all(x in device_name for x in ("RTX", "6000")):
        triple = f"turing-rtx6000-{system_os}"
    elif all(x in device_name for x in ("RTX", "8000")):
        triple = f"turing-rtx8000-{system_os}"
    elif all(x in device_name for x in ("TITAN", "RTX")):
        triple = f"turing-titanrtx-{system_os}"
    elif all(x in device_name for x in ("GTX", "1060")):
        triple = f"pascal-gtx1060-{system_os}"
    elif all(x in device_name for x in ("GTX", "1070")):
        triple = f"pascal-gtx1070-{system_os}"
    elif all(x in device_name for x in ("GTX", "1080")):
        triple = f"pascal-gtx1080-{system_os}"

    # Amd Targets
    # Linux: Radeon RX 7900 XTX
    # Windows: AMD Radeon RX 7900 XTX
    elif all(x in device_name for x in ("RX", "7900")):
        triple = f"rdna3-7900-{system_os}"
    elif any(x in device_name for x in ("AMD", "Radeon")):
        triple = f"rdna2-unknown-{system_os}"
    else:
        triple = None
    return triple


def get_vulkan_triple_flag(device_name="", extra_args=[]):
    for flag in extra_args:
        if "-iree-vulkan-target-triple=" in flag:
            print(f"Using target triple {flag.split('=')[1]}")
            return None

    if device_name == "" or device_name == [] or device_name is None:
        vulkan_device = get_vulkan_device_name()
    else:
        vulkan_device = device_name
    triple = get_vulkan_target_triple(vulkan_device)
    if triple is not None:
        print(
            f"Found vulkan device {vulkan_device}. Using target triple {triple}"
        )
        return f"-iree-vulkan-target-triple={triple}"
    print(
        """Optimized kernel for your target device is not added yet.
        Contact SHARK Admin on discord[https://discord.com/invite/RUqY2h2s9u]
        or pull up an issue."""
    )
    print(f"Target : {vulkan_device}")
    return None


def get_iree_vulkan_args(extra_args=[]):
    # vulkan_flag = ["--iree-flow-demote-i64-to-i32"]

    res_vulkan_flag = []
    vulkan_triple_flag = None
    for arg in extra_args:
        if "-iree-vulkan-target-triple=" in arg:
            print(f"Using target triple {arg} from command line args")
            vulkan_triple_flag = arg
            break

    if vulkan_triple_flag is None:
        vulkan_triple_flag = get_vulkan_triple_flag(extra_args=extra_args)

    if vulkan_triple_flag is not None:
        vulkan_target_env = get_vulkan_target_env_flag(vulkan_triple_flag)
        res_vulkan_flag.append(vulkan_target_env)
    return res_vulkan_flag


def set_iree_vulkan_runtime_flags(flags):
    for flag in flags:
        ireert.flags.parse_flags(flag)
    return
