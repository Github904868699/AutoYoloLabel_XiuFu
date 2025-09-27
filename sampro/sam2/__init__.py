# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra


def _initialize_hydra_configs() -> None:
    """Register the Hydra config package used by SAM2.

    The upstream project installs as a top-level ``sam2`` package, but inside
    AutoYOLO the module lives under ``sampro.sam2``.  We try both locations so
    that the code works whether users rely on the vendored copy or install the
    original wheel separately.
    """

    for module_name in ("sampro.sam2", "sam2"):
        try:
            initialize_config_module(module_name, version_base="1.2")
            return
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError(
        "未能找到 SAM2 的 Hydra 配置包，请确认项目完整或已正确安装 sam2。"
    )


if not GlobalHydra.instance().is_initialized():
    _initialize_hydra_configs()

# 提供一个与上游一致的顶层别名 ``sam2``，
# 以便 Hydra 配置中的 ``_target_: sam2.xxx`` 可以正常解析。
if "sam2" not in sys.modules and importlib.util.find_spec("sam2") is None:
    sys.modules["sam2"] = sys.modules[__name__]
