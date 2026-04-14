import torch


class StatsTracker:
    def __init__(self, name, ema, track_median=True):
        self.name = name
        self.ema = ema
        self.track_median = track_median
        self.n_updates = 0
        self.last_value = None
        self.ema_value = None
        self.avg_value = None
        self.median_value = None
        # P² algorithm state (Jain & Chlamtac, 1985).
        # Lazily initialized on the first update.
        # Scalar path: _p2_q is list[float](5), _p2_n is list[int](5).
        # Flat-tensor path: _p2_q is Tensor(5, N), _p2_n is Tensor(5, N, long).
        self._p2_q = None
        self._p2_n = None
        self._p2_dn = [0.0, 0.25, 0.5, 0.75, 1.0]  # desired position increments for p=0.5

    def _p2_update(self, x):
        N = self.n_updates
        is_tensor = isinstance(x, torch.Tensor) and x.ndim > 0

        # Bootstrap: collect the first 5 observations, computing the exact median each time.
        if N <= 5:
            if N == 1:
                if is_tensor:
                    n_elem = x.shape[0]
                    self._p2_q = x.new_zeros(5, n_elem, dtype=torch.float)
                    self._p2_n = x.new_zeros(5, n_elem, dtype=torch.long)
                else:
                    self._p2_q = [0.0] * 5
                    self._p2_n = [1, 2, 3, 4, 5]
            if is_tensor:
                self._p2_q[N - 1] = x.detach().float()
                sorted_q = self._p2_q[:N].sort(dim=0).values  # (N, n_elem)
            else:
                self._p2_q[N - 1] = float(x)
                sorted_q = sorted(self._p2_q[:N])
            mid = (N - 1) // 2
            if N % 2 == 1:
                self.median_value = sorted_q[mid].clone() if is_tensor else sorted_q[mid]
            else:
                mid_val = (sorted_q[mid] + sorted_q[mid + 1]) / 2
                self.median_value = mid_val.clone() if is_tensor else mid_val
            if N == 5:
                if is_tensor:
                    self._p2_q = sorted_q
                    n_elem = self._p2_q.shape[1]
                    self._p2_n = (
                        torch.arange(1, 6, dtype=torch.long, device=self._p2_q.device)
                        .view(5, 1).expand(5, n_elem).clone()
                    )
                else:
                    self._p2_q = list(sorted_q)
            return

        dn = self._p2_dn

        if is_tensor:
            xf = x.detach().float()      # (N,)
            q = self._p2_q               # (5, N)
            n = self._p2_n               # (5, N) long

            # Update extremes and find per-element bin k in [0, 3].
            q[0] = torch.minimum(q[0], xf)
            q[4] = torch.maximum(q[4], xf)
            k = (q <= xf.unsqueeze(0)).sum(dim=0).sub(1).clamp(0, 3)  # (N,)

            # Shift positions of markers to the right of k.
            for i in range(1, 5):
                n[i].add_((i > k).long())

            # Adjust the three middle markers via parabolic interpolation.
            for i in range(1, 4):
                d = (1.0 + dn[i] * (N - 1)) - n[i].float()     # (N,)
                mask = (
                    ((d >= 1.0) & (n[i + 1] - n[i] > 1)) |
                    ((d <= -1.0) & (n[i - 1] - n[i] < -1))
                )
                if not mask.any():
                    continue
                sign = d.sign()                                  # (N,) ±1
                ni      = n[i].float()
                ni_prev = n[i - 1].float()
                ni_next = n[i + 1].float()
                qi_prev = q[i - 1]
                qi_next = q[i + 1]

                # Parabolic formula.
                qi_new = q[i] + sign / (ni_next - ni_prev) * (
                    (ni - ni_prev + sign) * (qi_next - q[i]) / (ni_next - ni) +
                    (ni_next - ni - sign) * (q[i] - qi_prev) / (ni - ni_prev)
                )
                # Linear fallback where parabolic violates marker order.
                q_nbr = torch.where(sign > 0, qi_next, qi_prev)
                n_nbr = torch.where(sign > 0, ni_next, ni_prev)
                qi_lin = q[i] + sign * (q_nbr - q[i]) / (n_nbr - ni)
                qi_new = torch.where((qi_prev < qi_new) & (qi_new < qi_next), qi_new, qi_lin)

                q[i] = torch.where(mask, qi_new, q[i])
                n[i] = torch.where(mask, (ni + sign).long(), n[i])

            self.median_value = q[2].clone()

        else:
            xf = float(x)
            q = self._p2_q
            n = self._p2_n

            if xf < q[0]:
                q[0] = xf
                k = 0
            elif xf < q[1]:
                k = 0
            elif xf < q[2]:
                k = 1
            elif xf < q[3]:
                k = 2
            elif xf <= q[4]:
                k = 3
            else:
                q[4] = xf
                k = 3

            for i in range(k + 1, 5):
                n[i] += 1

            for i in range(1, 4):
                d = 1.0 + dn[i] * (N - 1) - n[i]
                if (d >= 1.0 and n[i + 1] - n[i] > 1) or (d <= -1.0 and n[i - 1] - n[i] < -1):
                    sign = 1 if d > 0 else -1
                    qi_new = q[i] + sign / (n[i + 1] - n[i - 1]) * (
                        (n[i] - n[i - 1] + sign) * (q[i + 1] - q[i]) / (n[i + 1] - n[i]) +
                        (n[i + 1] - n[i] - sign) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
                    )
                    if not (q[i - 1] < qi_new < q[i + 1]):
                        j = i + sign
                        qi_new = q[i] + sign * (q[j] - q[i]) / (n[j] - n[i])
                    q[i] = qi_new
                    n[i] += sign

            self.median_value = q[2]

    def update(self, value):
        self.n_updates += 1
        self.last_value = value.detach().clone() if isinstance(value, torch.Tensor) else value
        if self.n_updates == 1:
            self.ema_value = self.last_value
            self.avg_value = self.last_value
        else:
            self.ema_value = self.ema_value * self.ema + value * (1 - self.ema)
            self.avg_value = (self.avg_value * (self.n_updates - 1) + value) / self.n_updates
        if self.track_median:
            self._p2_update(value)

    def get(self):
        name = self.name
        result = {
            f"{name}": self.last_value,
            f"ema_{name}": self.ema_value,
            f"avg_{name}": self.avg_value,
        }
        if self.track_median:
            result[f"median_{name}"] = self.median_value
        return result

    def state_dict(self):
        p2_q = self._p2_q.clone() if isinstance(self._p2_q, torch.Tensor) else list(self._p2_q) if self._p2_q is not None else None
        p2_n = self._p2_n.clone() if isinstance(self._p2_n, torch.Tensor) else list(self._p2_n) if self._p2_n is not None else None
        return {
            "n_updates": self.n_updates,
            "value": self.last_value,
            "ema_value": self.ema_value,
            "avg_value": self.avg_value,
            "median_value": self.median_value,
            "p2_q": p2_q,
            "p2_n": p2_n,
        }

    def load_state_dict(self, state_dict):
        self.n_updates = state_dict["n_updates"]
        self.last_value = state_dict["value"]
        self.ema_value = state_dict["ema_value"]
        self.avg_value = state_dict["avg_value"]
        self.median_value = state_dict.get("median_value")
        p2_q = state_dict.get("p2_q")
        p2_n = state_dict.get("p2_n")
        # Support legacy checkpoints that stored p2_q/p2_n as plain lists.
        self._p2_q = list(p2_q) if isinstance(p2_q, list) else p2_q
        self._p2_n = list(p2_n) if isinstance(p2_n, list) else p2_n
