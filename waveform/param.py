"""FHSS使用到的所有参数定义：采样率、中心频率、每符号采样数、滚降系数等等等等

标准单位：频率 Hz， 时间 s ，速率 sps

---用法---
    from waveform.param import P    #默认配置
    fs = P.f_s                      # 只使用单一参数
    cfg = Params(f_chip=5.0e6, samples_per_chip=2)  #参数设置

"""


from __future__ import annotations

import math
import os
import warnings

# HackRf One 硬件限制最大采样率Msps
HackRF_MAX_FS = 20.0e6

@dataclass(frozen=True)
class Param:
    """链路中所有可调参数，派生量以属性（property）形式给出。

    默认值为**bring-up（调试/初步验证）**配置（2 Mcps）。
    5 Mcps 目标配置为 ``Params(f_chip=5.0e6, samples_per_chip=2)``。
    """



