import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class DeepTemporalModel(nn.Module):  # 输入 [B, 12, 12]，输出 [B, 2]
    def __init__(self, input_dim=12, hidden_dim=64, basis_dim=12, out_dim=2,
                 fixed_pv: torch.Tensor | None = None):
        """
        fixed_pv=None  -> pv 为可训练参数（trainable）
        fixed_pv=Tensor-> pv 为固定 buffer（fixed），形状可为 [12] 或 [1,12]
        """
        super().__init__()
        self.out_dim = out_dim
        self.basis_dim = basis_dim

        # 时序主干
        self.rnn_0  = nn.RNN(input_size=input_dim, hidden_size=32, batch_first=True)
        self.lstm_1 = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.gru_3  = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # 基展开到 [B, out_dim*basis_dim]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Linear(32, 16),
            nn.Linear(16, out_dim * basis_dim)
        )

        # cv 始终可训练
        self.cv = nn.Parameter(torch.ones(1, basis_dim))  # [1, 12]

        # pv：初始化即定型，不再切换
        if fixed_pv is None:
            self.pv = nn.Parameter(torch.ones(1, basis_dim))  # trainable
            self._pv_is_fixed = False
        else:
            if fixed_pv.dim() == 1:
                fixed_pv = fixed_pv.unsqueeze(0)
            assert fixed_pv.shape == (1, basis_dim), f"fixed_pv must be [1,{basis_dim}] or [{basis_dim}]"
            self.register_buffer("pv", fixed_pv.detach().clone())  # fixed
            self._pv_is_fixed = True

    def forward(self, x):  # x: [B, 12, 12]
        x, _ = self.rnn_0(x)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x, _ = self.gru_3(x)

        h_last = x[:, -1, :]                                   # [B, 64]
        y = self.mlp(h_last).view(-1, self.out_dim, self.basis_dim)  # [B, 2, 12]

        scale = (self.cv * self.pv).unsqueeze(1)               # [1,1,12] → 广播到 [B,2,12]
        y = y * scale
        return y.sum(dim=-1)                                   # [B, 2]

    # —— 只读工具方法（保留以便记录与打印） ——
    def pv_mode(self) -> str:
        return "fixed" if self._pv_is_fixed else "trainable"

    def pv_value(self) -> torch.Tensor:
        """返回当前 pv（tensor，随设备移动）。"""
        return self.pv


# ========== 与训练脚本一致的 pv 值 ==========
# sinD
pv_value_sinD=[[1.0047965, 1.0075866, 1.0141581, 0.9986545, 0.9934685, 1.0089078, 1.0054502,
  1.026036,  1.0024174, 0.9975765, 1.0100975, 0.992092 ]]

# inD
pv_value_inD=[[1.0011618,  0.9851395,  1.0112344,  1.0013374,  0.9937576,  0.9829733,
  0.989543,   1.0038168,  0.98837113, 0.99768245, 0.9988602,  0.9905746 ]]

# CitySim
pv_value_CitySim=[[1.0281078, 1.0291625, 1.0343446, 1.0068531, 1.0230848, 1.0137218, 1.0193505,
  1.0267537, 1.0026474, 1.0213368, 1.0249325, 1.0026276]]


def infer_pv_from_path(model_path: Path) -> str:
    """
    根据路径 .../<percent>/<pv_option>/<run>/model.pt 推断 pv_option
    """
    parts = model_path.parts
    # 期望结构: .../inD_trained_weights/<percent>/<pv_option>/<run>/xxx.pt
    # 安全起见从后往前找
    for i in range(len(parts)-1, -1, -1):
        if parts[i] in ("None", "sinD", "CitySim", "inD"):
            return parts[i]
    # 若没找到，默认 None
    return "None"

def load_model_with_pv(model_path: Path) -> DeepTemporalModel:
    """
    - 从路径推断 pv_option
    - 选择相应 pv tensor 注入模型
    - 加载权重（兼容 _orig_mod. 前缀）
    """
    pv_opt = infer_pv_from_path(model_path)
    pv_map = {
        "sinD": pv_value_sinD,
        "inD": pv_value_inD,
        "CitySim": pv_value_CitySim,
        "None": None
    }
    pv_value = pv_map.get(pv_opt, None)
    pv_tensor = torch.tensor(pv_value, dtype=torch.float32) if pv_value is not None else None

    if pv_tensor is not None:
        model = DeepTemporalModel(fixed_pv=pv_tensor)
        logger.info(f"[PV] Using pv_option={pv_opt}")
    else:
        model = DeepTemporalModel()
        logger.info("[PV] pv disabled (None)")

    sd = torch.load(model_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}

    # 某些版本可能包含 module. 前缀
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"[StateDict] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"[StateDict] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    logger.info(f": Loaded model weights from {model_path}")
    return model

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False