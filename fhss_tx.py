#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FHSS 跳频通信演示系统 —— 发送端 (HackRF + GNU Radio 3.10 + SoapySDR)
====================================================================

信号链:
    随机字节源 ──┬─► [显示] 源比特 NRZ 时域波形
                 │
                 └─► QPSK 调制 (generic_mod, 含 RRC 根升余弦脉冲整形)
                       ├─► [显示] QPSK 基带时域波形 (I/Q)
                       ├─► [显示] 跳频前频谱
                       └─► FHSS 跳频器 (连续相位, PN 跳频序列)
                             ├─► [显示] 跳频后时域波形
                             ├─► [显示] 跳频后频谱
                             ├─► [显示] 瀑布图
                             └─► HackRF 发送 (soapy.sink)

GUI 控制:
    * TX VGA(IF) 增益滑块  0~47 dB   —— HackRF 发送主功率
    * RF AMP    增益滑块  0/14 dB    —— HackRF 射频放大器(开/关)
    * 跳频速率  滑块                 —— 调整每跳驻留时间
    * 跳频 开/关 按钮

说明:
    HackRF 的 LNA 仅在“接收”链路有效(0~40dB)，发送链路没有 LNA，
    发送端可调增益为 VGA(IF) 与 AMP。LNA 滑块将放在接收端程序中。

本地环路演示: 用 SMA 线把本机(TX)与另一台 HackRF(RX)对接即可。
"""

import sys
import os
import signal
import struct
import zlib
import numpy as np

from PyQt5 import Qt, QtCore, QtWidgets

from gnuradio import gr, blocks, analog, digital, qtgui, soapy, fec
from gnuradio.fft import window

try:
    import sip
except ImportError:
    from PyQt5 import sip


# =========================================================================
#  全局参数
# =========================================================================
SAMP_RATE   = 2_000_000          # HackRF 采样率 / 瞬时带宽 (Hz)
CENTER_FREQ = 915_000_000        # 射频中心频率 (Hz, ISM 频段)
SPS         = 16                 # 每符号采样数 -> 符号率 = 125 kbaud
EXCESS_BW   = 0.35               # RRC 滚降系数 (脉冲整形)

NUM_HOPS     = 8                 # 跳频信道数
CHAN_SPACING = 200_000           # 信道间隔 (Hz)  -> 占用 ±700 kHz
DEFAULT_HOP_RATE = 1000          # 默认跳频速率 (hops/s)

# 选择用于发送的 HackRF (本机有两台，按需修改序列号；留空则用第一台)
TX_SERIAL = "000000000000000026a468dc210c8da7"

# 初始增益 (保守值，避免直连接收端过载)
INIT_VGA = 20                    # 0~47 dB
INIT_AMP = 0                     # 0 或 14 dB

# 数字基带幅度归一化 (generic_mod 输出 RMS≈2，需缩放到 |x|<1 以免 HackRF 削波)
TX_AMPLITUDE = 0.15

# ---- LDPC 信道编码 ----
# 采用 GNU Radio 自带的 LDPC 校验矩阵 (K=902 信息位, N=1800 码字位, 码率≈1/2)
# N 可被 8 整除, 便于 比特<->字节 对齐
LDPC_ALIST = "/usr/share/gnuradio/fec/ldpc/n_1800_k_0902_gap_28.alist"
LDPC_GAP   = 28

# ---- 帧结构 (信息位域, 每帧 K 比特) ----
# 每帧 = K//8 字节(余数补零). 字节布局:
#   [0]      帧类型   [1:3] 序号(uint16)  [3] 本帧有效载荷长度
#   [4:4+P]  载荷     [尾部4字节] CRC32(覆盖前面所有字节)
FRAME_SYNC   = 0x53   # 'S' 同步帧(固定已知图案, 供接收端相关同步)
FRAME_HEADER = 0x48   # 'H' 文件头(文件大小/帧数/文件名)
FRAME_DATA   = 0x44   # 'D' 数据帧
FRAME_IDLE   = 0x49   # 'I' 空闲帧(无文件时填充, 维持载波与跳频)


# =========================================================================
#  FHSS 跳频器 (自定义同步块)
#  将基带信号乘以一个“连续相位”的载波，载波频率按 PN 序列每隔
#  dwell 个采样跳变一次。连续相位可显著减小跳变时的频谱泄漏。
# =========================================================================
class fhss_hopper(gr.sync_block):
    def __init__(self, samp_rate, hop_freqs, dwell, enable=True, seed=42):
        gr.sync_block.__init__(
            self,
            name="FHSS Hopper",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.samp_rate = float(samp_rate)
        self.hop_freqs = list(hop_freqs)
        self.dwell = int(dwell)
        self.enable = bool(enable)

        # 用固定种子生成一个伪随机跳频序列(信道访问顺序)
        rng = np.random.RandomState(seed)
        self.hop_order = list(rng.permutation(len(self.hop_freqs)))

        self.phase = 0.0          # 连续相位累加器
        self.idx = 0              # 当前在 hop_order 中的位置
        self.in_hop = 0           # 当前跳已发出的采样数

    # --- 运行时可调参数 (GUI 回调) ---
    def set_enable(self, en):
        self.enable = bool(en)

    def set_dwell(self, dwell):
        self.dwell = max(1, int(dwell))

    def current_freq(self):
        return self.hop_freqs[self.hop_order[self.idx]]

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)

        # 关闭跳频: 直接旁路 (相当于驻留在中心频点)
        if not self.enable:
            out[:] = inp
            return n

        produced = 0
        two_pi = 2.0 * np.pi
        while produced < n:
            remain = self.dwell - self.in_hop
            take = min(remain, n - produced)

            f = self.hop_freqs[self.hop_order[self.idx]]
            inc = two_pi * f / self.samp_rate
            # 相位斜坡 (从当前相位继续累加，保证跳变处相位连续)
            k = np.arange(1, take + 1, dtype=np.float64)
            carrier = np.exp(1j * (self.phase + inc * k)).astype(np.complex64)

            seg = slice(produced, produced + take)
            out[seg] = inp[seg] * carrier

            self.phase = (self.phase + inc * take) % two_pi
            self.in_hop += take
            produced += take

            if self.in_hop >= self.dwell:
                self.in_hop = 0
                self.idx = (self.idx + 1) % len(self.hop_order)

        return n


# =========================================================================
#  发送数据源 (自定义源块)
#  输出 “信息比特流” (每字节 1 比特, 取值 0/1), 下游接 LDPC 编码器。
#  两种模式:
#    random : 连续随机比特 (用于频谱/瀑布演示)
#    file   : 把真实文件封装成 [SYNC][HEADER][DATA...] 帧序列后逐比特发出
#  通过 set_output_multiple(K) 保证每次 work 产生整数个 K 比特帧,
#  从而运行时切换模式/文件时, 下游 LDPC 编码器始终保持帧对齐。
# =========================================================================
class fhss_tx_source(gr.sync_block):
    def __init__(self, k_bits):
        gr.sync_block.__init__(
            self, name="FHSS TX Source",
            in_sig=None, out_sig=[np.uint8])

        self.K = int(k_bits)
        self.frame_bytes = self.K // 8          # 每帧承载的字节数(含头/CRC)
        self.pad_bits = self.K - self.frame_bytes * 8
        self.payload_len = self.frame_bytes - 1 - 2 - 1 - 4   # 去掉 类型/序号/长度/CRC

        self.set_output_multiple(self.K)        # 保证帧对齐

        self.lock_obj = __import__("threading").Lock()
        self.mode = "random"
        self.loop = True
        self.program = self._build_random()     # 当前要发送的比特序列
        self.pos = 0
        self.done = False                       # 文件单次发送完成标志
        self.status = "随机数据流"

    # ---------- 帧构造 ----------
    def _make_frame(self, ftype, seq, payload=b""):
        payload = payload[:self.payload_len]
        plen = len(payload)
        body = bytes([ftype]) + struct.pack(">H", seq & 0xFFFF) + bytes([plen])
        body += payload + b"\x00" * (self.payload_len - plen)   # 载荷补零
        crc = zlib.crc32(body) & 0xFFFFFFFF
        frame = body + struct.pack(">I", crc)                   # = frame_bytes 字节
        bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
        if self.pad_bits:
            bits = np.concatenate([bits, np.zeros(self.pad_bits, np.uint8)])
        return bits                                             # 长度 = K

    def _make_sync(self):
        # 固定已知图案(0xA5 序列), 接收端据此做帧/跳频同步
        patt = bytes([0xA5]) * self.payload_len
        return self._make_frame(FRAME_SYNC, 0, patt)

    def _build_random(self):
        # 64 帧随机比特, 循环发送
        return np.random.randint(0, 2, 64 * self.K).astype(np.uint8)

    def _build_idle(self):
        return np.concatenate([self._make_sync() for _ in range(4)])

    def _build_file(self, path):
        with open(path, "rb") as f:
            data = f.read()
        name = os.path.basename(path).encode("utf-8")[:96]
        nchunks = (len(data) + self.payload_len - 1) // self.payload_len
        frames = [self._make_sync()]
        hdr = struct.pack(">I", len(data)) + struct.pack(">I", nchunks) + name
        frames.append(self._make_frame(FRAME_HEADER, 0, hdr))
        for i in range(nchunks):
            chunk = data[i * self.payload_len:(i + 1) * self.payload_len]
            frames.append(self._make_frame(FRAME_DATA, i, chunk))
        self._file_info = (path, len(data), nchunks)
        return np.concatenate(frames)

    # ---------- 运行时控制 ----------
    def set_random(self):
        with self.lock_obj:
            self.mode = "random"
            self.program = self._build_random()
            self.pos = 0
            self.done = False
            self.status = "随机数据流"

    def set_file(self, path, loop):
        with self.lock_obj:
            try:
                self.program = self._build_file(path)
            except Exception as e:
                self.status = "文件读取失败: %s" % e
                return False
            self.mode = "file"
            self.loop = bool(loop)
            self.pos = 0
            self.done = False
            p, sz, nc = self._file_info
            self.status = "发送文件: %s  (%d 字节, %d 数据帧, %s)" % (
                os.path.basename(p), sz, nc, "循环" if loop else "单次")
            return True

    def get_status(self):
        return self.status

    # ---------- 数据生成 ----------
    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)                 # 一定是 K 的整数倍
        with self.lock_obj:
            filled = 0
            while filled < n:
                prog = self.program
                if self.pos >= len(prog):
                    # 一轮发送结束
                    if self.mode == "file" and not self.loop and not self.done:
                        self.done = True
                        self.program = self._build_idle()
                        self.status = "文件发送完成, 进入空闲(SYNC)填充"
                        self.pos = 0
                        prog = self.program
                    else:
                        self.pos = 0                 # 循环
                take = min(len(prog) - self.pos, n - filled)
                out[filled:filled + take] = prog[self.pos:self.pos + take]
                self.pos += take
                filled += take
        return n


# =========================================================================
#  顶层流图 + GUI
# =========================================================================
class fhss_tx(gr.top_block, QtWidgets.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "FHSS Transmitter")
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle("FHSS 跳频发送端演示 (HackRF)")

        # 跳频信道频率(基带, 以 0 为中心)
        self.hop_freqs = ((np.arange(NUM_HOPS) - (NUM_HOPS - 1) / 2.0)
                          * CHAN_SPACING).tolist()
        self.dwell = int(SAMP_RATE / DEFAULT_HOP_RATE)

        self._build_flowgraph()
        self._build_gui()

    # ------------------------------------------------------------------
    def _build_flowgraph(self):
        # --- 0. LDPC 信道编码器 (K=902 -> N=1800, 码率≈1/2) ---
        self.ldpc_H = fec.ldpc_H_matrix(LDPC_ALIST, LDPC_GAP)
        self.ldpc_enc = fec.ldpc_par_mtrx_encoder_make_H(self.ldpc_H)
        self.K = self.ldpc_enc.get_input_size()
        self.N = self.ldpc_enc.get_output_size()

        # --- 1. 数据源: 随机比特 / 文件帧 (输出信息比特, 每字节 1 比特) ---
        self.src = fhss_tx_source(self.K)
        self.encoder = fec.extended_encoder(self.ldpc_enc, None, "11")  # 信息位->码字位
        self.pack = blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST)    # 码字位->打包字节

        # --- 2. QPSK 调制 (含 RRC 脉冲整形) ---
        qpsk = digital.constellation_qpsk().base()
        self.mod = digital.generic_mod(
            constellation=qpsk,
            differential=True,
            samples_per_symbol=SPS,
            pre_diff_code=True,
            excess_bw=EXCESS_BW,
            verbose=False,
            log=False,
        )
        # 幅度归一化, 防止 HackRF (fc32, |x|<1) 削波
        self.amp = blocks.multiply_const_cc(TX_AMPLITUDE)

        # --- 3. 源比特 NRZ 显示支路 (显示进入编码器的信息比特) ---
        self.b2f = blocks.uchar_to_float()

        # --- 4. FHSS 跳频器 ---
        self.hopper = fhss_hopper(SAMP_RATE, self.hop_freqs,
                                  self.dwell, enable=True)

        # --- 5. HackRF 发送 (soapy.sink) ---
        dev = "driver=hackrf"
        if TX_SERIAL:
            dev += ",serial=" + TX_SERIAL
        self.sink = soapy.sink(dev, "fc32", 1, "", "", [""], [""])
        self.sink.set_sample_rate(0, SAMP_RATE)
        self.sink.set_bandwidth(0, 0)           # 0 = 自动
        self.sink.set_frequency(0, CENTER_FREQ)
        self.sink.set_gain(0, "VGA", float(INIT_VGA))
        self.sink.set_gain(0, "AMP", float(INIT_AMP))

        # --- 6. 显示用 qtgui sink ---
        self.ts_src   = self._mk_time_f("① 发送信息比特 (LDPC编码前)")
        self.ts_qpsk  = self._mk_time_c("② QPSK 调制后基带 (I/Q)")
        self.ts_hop   = self._mk_time_c("③ 跳频后时域波形 (I/Q)")
        self.fs_pre   = self._mk_freq("④ 跳频前 频谱")
        self.fs_post  = self._mk_freq("⑤ 跳频后 频谱")
        self.wf       = self._mk_waterfall("⑥ 跳频瀑布图")

        # --- 连接 ---
        # 信息比特 -> LDPC 编码 -> 打包字节 -> QPSK 调制 -> 幅度归一化
        self.connect(self.src, self.encoder, self.pack, self.mod, self.amp)
        # 信息比特 -> NRZ 显示
        self.connect(self.src, self.b2f, self.ts_src)

        # 归一化输出 -> 跳频器 / QPSK时域 / 跳频前频谱
        self.connect(self.amp, self.hopper)
        self.connect(self.amp, self.ts_qpsk)
        self.connect(self.amp, self.fs_pre)

        # 跳频输出 -> HackRF / 跳频后时域 / 跳频后频谱 / 瀑布图
        self.connect(self.hopper, self.sink)
        self.connect(self.hopper, self.ts_hop)
        self.connect(self.hopper, self.fs_post)
        self.connect(self.hopper, self.wf)

    # ------------------------------------------------------------------
    #  qtgui sink 工厂方法 (用 try/except 兼容不同子版本的可选设置)
    # ------------------------------------------------------------------
    def _mk_time_f(self, title):
        s = qtgui.time_sink_f(1024, SAMP_RATE, title, 1, None)
        self._cfg_time(s, ymin=-0.5, ymax=1.5)
        return s

    def _mk_time_c(self, title):
        s = qtgui.time_sink_c(1024, SAMP_RATE, title, 1, None)
        self._cfg_time(s, ymin=-0.6, ymax=0.6)
        return s

    def _cfg_time(self, s, ymin, ymax):
        for fn, args in [("set_update_time", (0.05,)),
                         ("set_y_axis", (ymin, ymax)),
                         ("enable_grid", (True,)),
                         ("enable_autoscale", (False,)),
                         ("enable_tags", (False,))]:
            try:
                getattr(s, fn)(*args)
            except Exception:
                pass

    def _mk_freq(self, title):
        s = qtgui.freq_sink_c(1024, window.WIN_BLACKMAN_hARRIS,
                              0, SAMP_RATE, title, 1, None)
        for fn, args in [("set_update_time", (0.05,)),
                         ("set_y_axis", (-120, 0)),
                         ("enable_grid", (True,)),
                         ("enable_autoscale", (False,))]:
            try:
                getattr(s, fn)(*args)
            except Exception:
                pass
        return s

    def _mk_waterfall(self, title):
        s = qtgui.waterfall_sink_c(1024, window.WIN_BLACKMAN_hARRIS,
                                   0, SAMP_RATE, title, 1, None)
        for fn, args in [("set_update_time", (0.05,)),
                         ("set_intensity_range", (-120, 0)),
                         ("enable_grid", (True,))]:
            try:
                getattr(s, fn)(*args)
            except Exception:
                pass
        return s

    def _wrap(self, sink):
        return sip.wrapinstance(sink.qwidget(), QtWidgets.QWidget)

    # ------------------------------------------------------------------
    #  GUI 布局
    # ------------------------------------------------------------------
    def _build_gui(self):
        grid = QtWidgets.QGridLayout(self)

        # 第一行: 三个时域图
        grid.addWidget(self._wrap(self.ts_src),  0, 0)
        grid.addWidget(self._wrap(self.ts_qpsk), 0, 1)
        grid.addWidget(self._wrap(self.ts_hop),  0, 2)
        # 第二行: 两个频谱 + 瀑布图
        grid.addWidget(self._wrap(self.fs_pre),  1, 0)
        grid.addWidget(self._wrap(self.fs_post), 1, 1)
        grid.addWidget(self._wrap(self.wf),      1, 2)
        # 第三行: 文件传输控制
        grid.addWidget(self._build_file_controls(), 2, 0, 1, 3)
        # 第四行: 射频/跳频控制
        grid.addWidget(self._build_controls(),      3, 0, 1, 3)

        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 0)
        self.resize(1500, 1000)

    def _build_file_controls(self):
        box = QtWidgets.QGroupBox(
            "文件传输 (LDPC 信道编码, K=%d N=%d 码率≈1/2, 每帧载荷 %d 字节)"
            % (self.K, self.N, self.src.payload_len))
        h = QtWidgets.QHBoxLayout(box)

        self.btn_random = QtWidgets.QPushButton("随机数据流")
        self.btn_random.setMinimumHeight(40)
        self.btn_random.clicked.connect(self._on_random)

        self.btn_file = QtWidgets.QPushButton("选择文件并发送…")
        self.btn_file.setMinimumHeight(40)
        self.btn_file.clicked.connect(self._on_choose_file)

        self.chk_loop = QtWidgets.QCheckBox("循环发送")
        self.chk_loop.setChecked(True)

        self.lbl_status = QtWidgets.QLabel(self.src.get_status())
        self.lbl_status.setStyleSheet("font-weight:bold;color:#1565c0;")

        h.addWidget(self.btn_random)
        h.addWidget(self.btn_file)
        h.addWidget(self.chk_loop)
        h.addWidget(self.lbl_status, 1)

        # 定时刷新状态(文件发送完成等)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.timeout.connect(
            lambda: self.lbl_status.setText(self.src.get_status()))
        self._status_timer.start(300)
        return box

    def _on_random(self):
        self.src.set_random()

    def _on_choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择要发送的文件")
        if path:
            self.src.set_file(path, self.chk_loop.isChecked())

    def _build_controls(self):
        box = QtWidgets.QGroupBox("发送参数控制")
        h = QtWidgets.QHBoxLayout(box)

        # --- VGA (IF) 增益滑块 ---
        self.lbl_vga = QtWidgets.QLabel()
        self.sld_vga = self._slider(0, 47, INIT_VGA, self._on_vga)
        h.addLayout(self._slider_col("TX VGA(IF) 增益", self.sld_vga, self.lbl_vga))

        # --- AMP 增益滑块 (0/14) ---
        self.lbl_amp = QtWidgets.QLabel()
        self.sld_amp = self._slider(0, 14, INIT_AMP, self._on_amp)
        self.sld_amp.setTickInterval(14)
        h.addLayout(self._slider_col("RF AMP 放大器", self.sld_amp, self.lbl_amp))

        # --- 跳频速率滑块 ---
        self.lbl_rate = QtWidgets.QLabel()
        self.sld_rate = self._slider(100, 5000, DEFAULT_HOP_RATE, self._on_rate)
        h.addLayout(self._slider_col("跳频速率 (hops/s)", self.sld_rate, self.lbl_rate))

        # --- 跳频开关按钮 ---
        vb = QtWidgets.QVBoxLayout()
        self.btn_hop = QtWidgets.QPushButton()
        self.btn_hop.setCheckable(True)
        self.btn_hop.setChecked(True)
        self.btn_hop.setMinimumHeight(60)
        self.btn_hop.toggled.connect(self._on_hop)
        vb.addWidget(QtWidgets.QLabel("跳频控制"))
        vb.addWidget(self.btn_hop)
        h.addLayout(vb)

        # 触发一次回调，刷新所有标签
        self._on_vga(INIT_VGA)
        self._on_amp(INIT_AMP)
        self._on_rate(DEFAULT_HOP_RATE)
        self._on_hop(True)
        return box

    def _slider(self, lo, hi, val, cb):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setMinimum(lo)
        s.setMaximum(hi)
        s.setValue(val)
        s.setMinimumWidth(220)
        s.valueChanged.connect(cb)
        return s

    def _slider_col(self, title, slider, label):
        v = QtWidgets.QVBoxLayout()
        v.addWidget(QtWidgets.QLabel(title))
        v.addWidget(slider)
        v.addWidget(label)
        return v

    # ------------------------------------------------------------------
    #  GUI 回调
    # ------------------------------------------------------------------
    def _on_vga(self, val):
        self.sink.set_gain(0, "VGA", float(val))
        self.lbl_vga.setText("当前: %d dB" % val)

    def _on_amp(self, val):
        # AMP 只有 开(14) / 关(0) 两种状态，滑块自动吸附
        amp = 14 if val >= 7 else 0
        if amp != val:
            self.sld_amp.blockSignals(True)
            self.sld_amp.setValue(amp)
            self.sld_amp.blockSignals(False)
        self.sink.set_gain(0, "AMP", float(amp))
        self.lbl_amp.setText("当前: %s (%d dB)" % ("开" if amp else "关", amp))

    def _on_rate(self, hops):
        dwell = max(1, int(SAMP_RATE / hops))
        self.hopper.set_dwell(dwell)
        self.lbl_rate.setText("%d hops/s  (驻留 %d 采样)" % (hops, dwell))

    def _on_hop(self, checked):
        self.hopper.set_enable(checked)
        self.btn_hop.setText("跳频: 开" if checked else "跳频: 关")


# =========================================================================
def main():
    qapp = QtWidgets.QApplication(sys.argv)

    tb = fhss_tx()
    tb.start()
    tb.show()

    print("=" * 60)
    print("FHSS 发送端已启动")
    print("  中心频率 : %.3f MHz" % (CENTER_FREQ / 1e6))
    print("  采样率   : %.1f MSps" % (SAMP_RATE / 1e6))
    print("  符号率   : %.1f kBd (sps=%d)" % (SAMP_RATE / SPS / 1e3, SPS))
    print("  跳频信道 : %d 个, 间隔 %.0f kHz" % (NUM_HOPS, CHAN_SPACING / 1e3))
    print("  跳频范围 : %.0f ~ %+.0f kHz (基带)"
          % (min(tb.hop_freqs) / 1e3, max(tb.hop_freqs) / 1e3))
    print("=" * 60)

    def sig_handler(sig, frame):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # 让 Python 解释器周期性运行, 以便 Ctrl+C 生效
    timer = QtCore.QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == "__main__":
    main()
