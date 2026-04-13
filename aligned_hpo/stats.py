import torch


class StatsTracker:
    def __init__(self, name, ema):
        self.name = name
        self.ema = ema
        self.n_updates = 0
        self.last_value = None
        self.ema_value = None
        self.avg_value = None

    def update(self, value):
        self.n_updates += 1
        self.last_value = value.detach().clone() if isinstance(value, torch.Tensor) else value
        if self.n_updates == 1:
            self.ema_value = self.last_value
            self.avg_value = self.last_value
        else:
            self.ema_value = self.ema_value * self.ema + value * (1 - self.ema)
            self.avg_value = (self.avg_value * (self.n_updates - 1) + value) / self.n_updates

    def get(self):
        name = self.name
        return {
            f"{name}": self.last_value,
            f"ema_{name}": self.ema_value,
            f"avg_{name}": self.avg_value
        }

    def state_dict(self):
        return {
            "n_updates": self.n_updates,
            "value": self.last_value,
            "ema_value": self.ema_value,
            "avg_value": self.avg_value,
        }

    def load_state_dict(self, state_dict):
        self.n_updates = state_dict["n_updates"]
        self.last_value = state_dict["value"]
        self.ema_value = state_dict["ema_value"]
        self.avg_value = state_dict["avg_value"]
