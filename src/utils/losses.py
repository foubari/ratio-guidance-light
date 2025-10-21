"""
Density-ratio learning losses for PMI estimation.
Implements 5 different loss functions: Discriminator, DV, uLSIF, RuLSIF, KLIEP.

Based on the original ratio_guidance implementation but simplified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class DensityRatioLoss(nn.Module):
    """
    Comprehensive density-ratio learning losses for a scalar scorer T(x,y,t).

    Supported loss_type:
      - "disc"    : Discriminator (logistic/JSD). Optimal logit ~ log(q/r)
      - "dv"      : Donsker-Varadhan. Maximizes MI lower bound E_q[T] - log(E_r[exp(T)])
      - "ulsif"   : uLSIF (unconstrained Least-Squares). Optimal ratio w ~ q/r
      - "rulsif"  : Relative uLSIF with alpha-divergence
      - "kliep"   : KLIEP (KL). Optimal ratio w ~ q/r

    API:
      forward(T_real, T_fake) -> (loss, metrics)
        T_real: (B,) = T(x_i, y_i) on true joint q(x,y)
        T_fake: (B,) = T(x'_j, y'_j) on reference r(x,y)=p(x)p(y)
    """

    def __init__(
        self,
        loss_type: str = "disc",
        # General settings
        use_exp_w: bool = False,           # if True, w = exp(T); else w = softplus(T)
        # DV settings
        dv_use_ema: bool = True,           # use EMA for log normalizer
        dv_ema_rate: float = 0.99,
        # KLIEP settings
        kliep_lambda: float = 1.0,         # weight for normalization penalty
        kliep_temperature: float = 1.0,
        kliep_bound_A: float = None,
        # uLSIF settings
        ulsif_l2: float = 0.0,
        # RuLSIF settings
        rulsif_alpha: float = 0.2,
        rulsif_link: str = "exp",          # 'exp' | 'softplus' | 'identity'
        rulsif_norm_penalty: float = 0.0,
        rulsif_softplus_eps: float = 1e-6
    ):
        super().__init__()
        assert loss_type in {"disc", "dv", "ulsif", "rulsif", "kliep"}
        self.loss_type = loss_type
        self.use_exp_w = use_exp_w

        # DV parameters
        self.dv_use_ema = dv_use_ema
        self.dv_ema_rate = dv_ema_rate
        if loss_type == "dv":
            self.register_buffer('dv_ema_exp_t', torch.tensor(1.0))
            self.register_buffer('dv_ema_initialized', torch.tensor(False))

        # KLIEP parameters
        self.kliep_lambda = kliep_lambda
        self.kliep_temperature = float(kliep_temperature)
        self.kliep_bound_A = None if kliep_bound_A is None else float(kliep_bound_A)

        # uLSIF parameters
        self.ulsif_l2 = ulsif_l2

        # RuLSIF parameters
        self.rulsif_alpha = float(rulsif_alpha)
        assert 0.0 < self.rulsif_alpha <= 1.0, "rulsif_alpha must be in (0,1]"
        assert rulsif_link in {"exp", "softplus", "identity"}
        self.rulsif_link = rulsif_link
        self.rulsif_norm_penalty = rulsif_norm_penalty
        self.rulsif_eps = float(rulsif_softplus_eps)

        # Discriminator uses BCEWithLogitsLoss
        if self.loss_type == "disc":
            self._bce = nn.BCEWithLogitsLoss()

    def _w_from_T(self, T):
        """Positive ratio estimate w >= 0"""
        if self.use_exp_w:
            return torch.exp(torch.clamp(T, max=40.0))
        else:
            return F.softplus(T) + EPS

    @staticmethod
    def log_mean_exp(x, dim=0):
        """Numerically stable log(mean(exp(x)))."""
        max_x = x.max(dim=dim, keepdim=True)[0] if dim is not None else x.max()
        z = (x - max_x).exp().mean(dim=dim)
        if dim is not None:
            return (z.log() + max_x.squeeze(dim)).squeeze()
        else:
            return z.log() + max_x

    def _apply_kliep_stabilizers(self, g: torch.Tensor) -> torch.Tensor:
        """Apply temperature and bounding to KLIEP scores"""
        if self.kliep_temperature != 1.0:
            g = g * self.kliep_temperature
        if self.kliep_bound_A is not None:
            A = self.kliep_bound_A
            g = A * torch.tanh(g / max(A, 1e-6))
        return g

    def _rulsif_link_map(self, t: torch.Tensor) -> torch.Tensor:
        """Map raw outputs to positive r_alpha for RuLSIF"""
        if self.rulsif_link == "exp":
            return torch.exp(torch.clamp(t, max=40.0))
        elif self.rulsif_link == "softplus":
            return F.softplus(t) + self.rulsif_eps
        else:  # 'identity'
            return t.clamp(min=self.rulsif_eps)

    def forward(self, T_real: torch.Tensor, T_fake: torch.Tensor):
        """
        Args:
            T_real: (B,) = T(x_i, y_i) on true joint q(x,y)
            T_fake: (B,) = T(x'_j, y'_j) on reference r(x,y)
        Returns:
            loss: scalar tensor to minimize
            metrics: dict of metrics for monitoring
        """
        assert T_real.dim() == 1 and T_fake.dim() == 1, "Pass 1-D tensors of scalar outputs"

        metrics = {}

        if self.loss_type == "disc":
            # Discriminator (BCE with logits)
            logits = torch.cat([T_real, T_fake], dim=0)
            targets = torch.cat([
                torch.ones_like(T_real),
                torch.zeros_like(T_fake)
            ], dim=0)
            loss = self._bce(logits, targets)

            # Compute accuracies
            with torch.no_grad():
                real_acc = (torch.sigmoid(T_real) > 0.5).float().mean()
                fake_acc = (torch.sigmoid(T_fake) < 0.5).float().mean()
                total_acc = 0.5 * (real_acc + fake_acc)

            metrics.update({
                "disc_loss": loss.detach().item(),
                "real_acc": real_acc.item(),
                "fake_acc": fake_acc.item(),
                "total_acc": total_acc.item(),
                "T_real_mean": T_real.mean().item(),
                "T_fake_mean": T_fake.mean().item(),
            })

        elif self.loss_type == "dv":
            # Donsker-Varadhan MI estimator
            # Clamp scores to prevent numerical overflow
            T_real_clamp = torch.clamp(T_real, min=-20.0, max=20.0)
            T_fake_clamp = torch.clamp(T_fake, min=-20.0, max=20.0)
            
            E_q_T = T_real_clamp.mean()

            # Compute log(E_r[exp(T)])
            if self.dv_use_ema and self.training:
                with torch.no_grad():
                    exp_t_fake = T_fake_clamp.exp().mean()
                    if not self.dv_ema_initialized:
                        self.dv_ema_exp_t = self.dv_ema_exp_t.to(exp_t_fake.device)
                        self.dv_ema_exp_t.copy_(exp_t_fake)
                        self.dv_ema_initialized.fill_(True)
                    else:
                        self.dv_ema_exp_t.mul_(self.dv_ema_rate).add_(
                            (1 - self.dv_ema_rate) * exp_t_fake
                        )
                log_E_r_exp_T = self.dv_ema_exp_t.clamp(min=EPS).log()
            else:
                log_E_r_exp_T = self.log_mean_exp(T_fake_clamp)

            # DV bound (MI lower bound)
            dv_bound = E_q_T - log_E_r_exp_T

            # Loss: minimize negative bound
            loss = -dv_bound

            metrics.update({
                "dv_bound": dv_bound.detach().item(),
                "E_q_T": E_q_T.detach().item(),
                "log_E_r_exp_T": log_E_r_exp_T.detach().item(),
                "T_real_mean": T_real.mean().detach().item(),
                "T_real_std": T_real.std().detach().item() if T_real.numel() > 1 else 0,
                "T_fake_mean": T_fake.mean().detach().item(),
                "T_fake_std": T_fake.std().detach().item() if T_fake.numel() > 1 else 0,
            })

        elif self.loss_type == "ulsif":
            # uLSIF (unconstrained Least-Squares Importance Fitting)
            w_real = self._w_from_T(T_real)
            w_fake = self._w_from_T(T_fake)

            # Loss: 0.5 * E_r[w^2] - E_q[w]
            loss = 0.5 * torch.mean(w_fake ** 2) - torch.mean(w_real)

            if self.ulsif_l2 > 0.0:
                loss = loss + 0.5 * self.ulsif_l2 * (w_real.pow(2).mean() + w_fake.pow(2).mean())

            metrics.update({
                "ulsif_loss": loss.detach().item(),
                "E_q_w": w_real.mean().item(),
                "E_r_w": w_fake.mean().item(),
                "E_q_w2": (w_real ** 2).mean().item(),
                "E_r_w2": (w_fake ** 2).mean().item(),
                "w_real_std": w_real.std().item() if w_real.numel() > 1 else 0,
                "w_fake_std": w_fake.std().item() if w_fake.numel() > 1 else 0,
            })

        elif self.loss_type == "rulsif":
            # Relative uLSIF with alpha-divergence
            r_real = self._rulsif_link_map(T_real)
            r_fake = self._rulsif_link_map(T_fake)

            # E_mix[r^2] with mix = alpha*q + (1-alpha)*r
            Emix_r2 = self.rulsif_alpha * (r_real ** 2).mean() + \
                      (1.0 - self.rulsif_alpha) * (r_fake ** 2).mean()
            Eq_r = r_real.mean()

            # RuLSIF loss
            loss = 0.5 * Emix_r2 - Eq_r

            # Normalization penalty
            if self.rulsif_norm_penalty > 0.0:
                Emix_r = self.rulsif_alpha * r_real.mean() + (1.0 - self.rulsif_alpha) * r_fake.mean()
                loss = loss + self.rulsif_norm_penalty * (Emix_r - 1.0) ** 2

            # Compute true density ratio from r_alpha
            with torch.no_grad():
                r_real_clamped = r_real.clamp(max=(1.0 / self.rulsif_alpha) - 1e-6)
                r_fake_clamped = r_fake.clamp(max=(1.0 / self.rulsif_alpha) - 1e-6)

                true_r_real = (1 - self.rulsif_alpha) * r_real_clamped / \
                             (1 - self.rulsif_alpha * r_real_clamped).clamp(min=EPS)
                true_r_fake = (1 - self.rulsif_alpha) * r_fake_clamped / \
                             (1 - self.rulsif_alpha * r_fake_clamped).clamp(min=EPS)

            metrics.update({
                "rulsif_loss": loss.detach().item(),
                "E_q_r_alpha": r_real.mean().item(),
                "E_r_r_alpha": r_fake.mean().item(),
                "E_mix_r2": Emix_r2.item(),
                "alpha": self.rulsif_alpha,
                "true_r_real_mean": true_r_real.mean().item(),
                "true_r_fake_mean": true_r_fake.mean().item(),
            })

        elif self.loss_type == "kliep":
            # KLIEP with stabilizers
            g_real = self._apply_kliep_stabilizers(T_real)
            g_fake = self._apply_kliep_stabilizers(T_fake)

            # Core KLIEP loss: -E_q[log w] with constraint E_r[w] = 1
            if self.use_exp_w:
                Eq_logw = g_real.mean()
                w_fake = torch.exp(torch.clamp(g_fake, max=40.0))
            else:
                w_real = F.softplus(g_real) + EPS
                w_fake = F.softplus(g_fake) + EPS
                Eq_logw = torch.log(w_real).mean()

            Er_w = w_fake.mean()

            # Version 1: -E_q[log w] + λ(E_r[w] - 1)^2
            loss = -Eq_logw + self.kliep_lambda * (Er_w - 1.0) ** 2

            metrics.update({
                "kliep_loss": loss.detach().item(),
                "E_q_logw": Eq_logw.item(),
                "E_r_w": Er_w.item(),
                "constraint_resid": (Er_w - 1.0).abs().item(),
                "temperature": self.kliep_temperature,
                "g_real_mean": g_real.mean().item(),
                "g_fake_mean": g_fake.mean().item(),
            })

        return loss, metrics


if __name__ == "__main__":
    print("Testing density-ratio losses...")

    batch_size = 32

    # Test each loss type
    for loss_type in ["disc", "dv", "ulsif", "rulsif", "kliep"]:
        print(f"\nTesting {loss_type}:")

        loss_fn = DensityRatioLoss(loss_type=loss_type)

        # Simulate scores
        T_real = torch.randn(batch_size)  # Scores on real pairs
        T_fake = torch.randn(batch_size)  # Scores on fake pairs

        loss, metrics = loss_fn(T_real, T_fake)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics: {list(metrics.keys())}")

    print("\nAll tests passed!")
