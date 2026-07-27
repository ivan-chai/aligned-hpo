import numpy as np
import torch
from datasketches import kll_floats_sketch, vector_of_kll_floats_sketches


class StatsTracker:
    def __init__(self, name, ema, track_median=True, kll_k=200):
        self.name = name
        self.ema = ema
        self.track_median = track_median
        self.kll_k = kll_k
        self.n_updates = 0
        self.last_value = None
        self.ema_value = None
        self.avg_value = None
        self._kll = None  # kll_floats_sketch or vector_of_kll_floats_sketches

    def _kll_update(self, x):
        is_tensor = isinstance(x, torch.Tensor) and x.ndim > 0

        if is_tensor:
            xf = x.detach().float().cpu().numpy()  # (n_elem,)
            if self._kll is None:
                self._kll = vector_of_kll_floats_sketches(self.kll_k, xf.shape[0])
            self._kll.update(xf)
        else:
            if self._kll is None:
                self._kll = kll_floats_sketch(self.kll_k)
            self._kll.update(float(x))

    def update(self, value):
        if isinstance(value, torch.Tensor) and (self.n_updates > 1) and (value.device != self.last_value.device):
            device = value.device
            self.ema_value = self.ema_value.to(device)
            self.avg_value = self.avg_value.to(device)
        self.n_updates += 1
        self.last_value = value.detach().clone() if isinstance(value, torch.Tensor) else value
        if self.n_updates == 1:
            self.ema_value = self.last_value
            self.avg_value = self.last_value
        else:
            self.ema_value = self.ema_value * self.ema + value * (1 - self.ema)
            self.avg_value = (self.avg_value * (self.n_updates - 1) + value) / self.n_updates
        if self.track_median:
            self._kll_update(value)

    def get(self):
        name = self.name
        result = {
            f"{name}": self.last_value,
            f"ema_{name}": self.ema_value,
            f"avg_{name}": self.avg_value,
        }
        if self.track_median:
            if isinstance(self._kll, vector_of_kll_floats_sketches):
                median_np = self._kll.get_quantiles(0.5).squeeze(1).astype(np.float32)
                median = torch.from_numpy(median_np).to(self.last_value.device)
            else:
                median = self._kll.get_quantile(0.5)
            result[f"median_{name}"] = median
        return result

    def state_dict(self):
        if self._kll is None:
            kll_bytes = None
            kll_d = None
        elif isinstance(self._kll, vector_of_kll_floats_sketches):
            kll_bytes = self._kll.serialize()  # list of bytes, one per sketch
            kll_d = self._kll.d
        else:
            kll_bytes = self._kll.serialize()
            kll_d = None
        return {
            "n_updates": self.n_updates,
            "value": self.last_value,
            "ema_value": self.ema_value,
            "avg_value": self.avg_value,
            "kll": kll_bytes,
            "kll_d": kll_d,
        }

    def load_state_dict(self, state_dict):
        self.n_updates = state_dict["n_updates"]
        self.last_value = state_dict["value"]
        self.ema_value = state_dict["ema_value"]
        self.avg_value = state_dict["avg_value"]
        kll_bytes = state_dict.get("kll")
        kll_d = state_dict.get("kll_d")
        if kll_bytes is None:
            self._kll = None
        elif kll_d is not None:
            self._kll = vector_of_kll_floats_sketches(self.kll_k, kll_d)
            for i, b in enumerate(kll_bytes):
                self._kll.deserialize(b, i)
        else:
            self._kll = kll_floats_sketch.deserialize(kll_bytes)
