"""
Density-ratio learning losses for PMI estimation.
Implements 6 different loss functions: Discriminator, DV, uLSIF, RuLSIF, KLIEP, InfoNCE.

Based on the original ratio_guidance implementation but simplified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-3  # Increased to prevent gradient explosion when w→0


class DensityRatioLoss(nn.Module):
    """
    Comprehensive density-ratio learning losses for a scalar scorer T(x,y,t).

    Supported loss_type:
      - "disc"    : Discriminator (logistic/JSD). Optimal logit ~ log(q/r)
      - "dv"      : Donsker-Varadhan. Maximizes MI lower bound E_q[T] - log(E_r[exp(T)])
      - "ulsif"   : uLSIF (unconstrained Least-Squares). Optimal ratio w ~ q/r
      - "rulsif"  : Relative uLSIF with alpha-divergence
      - "kliep"   : KLIEP (KL). Optimal ratio w ~ q/r
      - "infonce" : Symmetric InfoNCE. Learns PMI = log(q(x,y)/(q(x)q(y))) + const
                    WARNING: Valid only in-domain (test marginals = train marginals)

    API:
      forward(T_real, T_fake) -> (loss, metrics)
        Standard methods (disc/dv/ulsif/rulsif/kliep):
          T_real: (B,) = T(x_i, y_i) on true joint q(x,y)
          T_fake: (B,) = T(x'_j, y'_j) on reference r(x,y)=p(x)p(y)

        InfoNCE only:
          T_real: (B, B) = T_mat where T_mat[i,j] = T(x_i, y_j, t)
          T_fake: ignored (can be None)
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
        rulsif_softplus_eps: float = 1e-6,
        # InfoNCE settings
        infonce_tau: float = 0.07          # temperature for InfoNCE
    ):
        super().__init__()
        assert loss_type in {"disc", "dv", "ulsif", "rulsif", "kliep", "infonce"}
        self.loss_type = loss_type
        self.use_exp_w = use_exp_w

        # InfoNCE parameters
        self.infonce_tau = infonce_tau

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
        """
        Map raw outputs to w_α ∈ (0, 1/α] for RuLSIF.
        w_α = q / (α*q + (1-α)*r)
        Default: softplus + clamp for stability
        """
        # Apply link function
        if self.rulsif_link == "exp":
            w_alpha = torch.exp(torch.clamp(t, max=40.0))
        elif self.rulsif_link == "softplus":
            w_alpha = F.softplus(t) + EPS  # Use EPS=1e-3
        else:  # 'identity'
            w_alpha = t.clamp(min=EPS)

        # Enforce theoretical bound: w_α < 1/α
        max_val = (1.0 / self.rulsif_alpha) - EPS
        w_alpha = w_alpha.clamp(max=max_val)

        return w_alpha

    def forward(self, T_real: torch.Tensor, T_fake: torch.Tensor = None):
        """
        Args:
            T_real: For standard methods: (B,) = T(x_i, y_i) on true joint q(x,y)
                    For InfoNCE: (B, B) = T_mat where T_mat[i,j] = T(x_i, y_j, t)
            T_fake: For standard methods: (B,) = T(x'_j, y'_j) on reference r(x,y)
                    For InfoNCE: ignored (can be None)
        Returns:
            loss: scalar tensor to minimize
            metrics: dict of metrics for monitoring
        """
        metrics = {}

        # InfoNCE has special input format: T_real is a (B, B) matrix
        if self.loss_type == "infonce":
            assert T_real.dim() == 2 and T_real.size(0) == T_real.size(1), \
                "InfoNCE requires T_real to be (B, B) matrix"

            # Symmetric InfoNCE loss
            # Formula: -1/B Σ_i [log(exp(S_ii) / Σ_j exp(S_ij)) + log(exp(S_ii) / Σ_j exp(S_ji))]
            # where S_ij = T(x_i, y_j, t) / τ

            B = T_real.size(0)
            S = T_real / self.infonce_tau  # (B, B) scaled scores

            # Row-wise softmax: log p(i|x_i) for each row i
            # log(exp(S_ii) / Σ_j exp(S_ij)) = S_ii - logsumexp_j(S_ij)
            log_prob_rows = S.diagonal() - torch.logsumexp(S, dim=1)  # (B,)

            # Column-wise softmax: log p(i|y_i) for each column i
            # log(exp(S_ii) / Σ_j exp(S_ji)) = S_ii - logsumexp_j(S_ji)
            log_prob_cols = S.diagonal() - torch.logsumexp(S, dim=0)  # (B,)

            # Symmetric InfoNCE: average both directions
            loss = -(log_prob_rows.mean() + log_prob_cols.mean()) / 2.0

            # Metrics
            with torch.no_grad():
                diag_scores = S.diagonal().mean().item()
                off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
                off_diag_scores = S[off_diag_mask].mean().item()
                accuracy_rows = (S.argmax(dim=1) == torch.arange(B, device=S.device)).float().mean().item()
                accuracy_cols = (S.argmax(dim=0) == torch.arange(B, device=S.device)).float().mean().item()

            metrics.update({
                "infonce_loss": loss.detach().item(),
                "tau": self.infonce_tau,
                "S_diag_mean": diag_scores,
                "S_offdiag_mean": off_diag_scores,
                "diag_vs_offdiag": diag_scores - off_diag_scores,
                "accuracy_rows": accuracy_rows,
                "accuracy_cols": accuracy_cols,
                "accuracy_avg": (accuracy_rows + accuracy_cols) / 2.0,
            })

            return loss, metrics

        # Standard methods: check for 1-D tensors
        assert T_real.dim() == 1 and T_fake.dim() == 1, "Pass 1-D tensors of scalar outputs"

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
            # Formula: J_DV(T) = E_q[T] - log E_r[exp(T)]
            E_q_T = T_real.mean()

            # Compute log(E_r[exp(T)]) with mixed differentiable/EMA approach
            # Clip T before exp to prevent overflow
            T_fake_clipped = torch.clamp(T_fake, max=10.0)

            if self.dv_use_ema and self.training:
                # Differentiable term for gradient flow
                log_E_r_exp_T_diff = self.log_mean_exp(T_fake_clipped)

                # EMA term for stability (no gradient)
                with torch.no_grad():
                    exp_t_fake = T_fake_clipped.exp().mean()
                    if not self.dv_ema_initialized:
                        self.dv_ema_exp_t = self.dv_ema_exp_t.to(exp_t_fake.device)
                        self.dv_ema_exp_t.copy_(exp_t_fake)
                        self.dv_ema_initialized.fill_(True)
                    else:
                        self.dv_ema_exp_t.mul_(self.dv_ema_rate).add_(
                            (1 - self.dv_ema_rate) * exp_t_fake
                        )
                    log_E_r_exp_T_ema = self.dv_ema_exp_t.clamp(min=EPS).log()

                # Mix: (1-β)*differentiable + β*EMA for stability
                beta = self.dv_ema_rate
                log_E_r_exp_T = (1 - beta) * log_E_r_exp_T_diff + beta * log_E_r_exp_T_ema
            else:
                log_E_r_exp_T = self.log_mean_exp(T_fake_clipped)

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
            # Formula: J(w) = 0.5 E_r[w²] - E_q[w] + λ(E_r[w] - 1)²
            w_real = self._w_from_T(T_real)
            w_fake = self._w_from_T(T_fake)

            # Core uLSIF loss: 0.5 * E_r[w^2] - E_q[w]
            loss = 0.5 * torch.mean(w_fake ** 2) - torch.mean(w_real)

            # Calibration constraint: E_r[w] ≈ 1
            E_r_w = w_fake.mean()
            calibration_penalty = (E_r_w - 1.0) ** 2
            loss = loss + 0.1 * calibration_penalty  # λ=0.1 for normalization

            # Note: L2 regularization should be applied via weight_decay in optimizer
            # not on outputs. Removed incorrect L2 on w.

            metrics.update({
                "ulsif_loss": loss.detach().item(),
                "E_q_w": w_real.mean().item(),
                "E_r_w": E_r_w.item(),
                "E_q_w2": (w_real ** 2).mean().item(),
                "E_r_w2": (w_fake ** 2).mean().item(),
                "calibration_penalty": calibration_penalty.item(),
                "w_real_std": w_real.std().item() if w_real.numel() > 1 else 0,
                "w_fake_std": w_fake.std().item() if w_fake.numel() > 1 else 0,
            })

        elif self.loss_type == "rulsif":
            # Relative uLSIF with alpha-divergence
            # Formula: J(w_α) = 0.5 E_mix[w_α²] - E_q[w_α] + λ(E_mix[w_α] - 1)²
            # where w_α = q/(α*q + (1-α)*r) ∈ (0, 1/α]
            w_alpha_real = self._rulsif_link_map(T_real)
            w_alpha_fake = self._rulsif_link_map(T_fake)

            # E_mix[w_α^2] with mix = alpha*q + (1-alpha)*r
            Emix_w2 = self.rulsif_alpha * (w_alpha_real ** 2).mean() + \
                      (1.0 - self.rulsif_alpha) * (w_alpha_fake ** 2).mean()
            Eq_w = w_alpha_real.mean()

            # RuLSIF loss
            loss = 0.5 * Emix_w2 - Eq_w

            # Normalization penalty: E_mix[w_α] ≈ 1
            Emix_w = self.rulsif_alpha * w_alpha_real.mean() + \
                     (1.0 - self.rulsif_alpha) * w_alpha_fake.mean()
            normalization_penalty = (Emix_w - 1.0) ** 2

            # Always apply normalization for stability
            loss = loss + 0.1 * normalization_penalty

            # Convert w_α → q/r using exact formula: q/r = (1-α)w_α / (1 - αw_α)
            with torch.no_grad():
                # w_α already clamped to < 1/α in _rulsif_link_map
                denom_real = (1.0 - self.rulsif_alpha * w_alpha_real).clamp(min=EPS)
                denom_fake = (1.0 - self.rulsif_alpha * w_alpha_fake).clamp(min=EPS)

                ratio_real = (1.0 - self.rulsif_alpha) * w_alpha_real / denom_real
                ratio_fake = (1.0 - self.rulsif_alpha) * w_alpha_fake / denom_fake

            metrics.update({
                "rulsif_loss": loss.detach().item(),
                "E_q_w_alpha": w_alpha_real.mean().item(),
                "E_r_w_alpha": w_alpha_fake.mean().item(),
                "E_mix_w2": Emix_w2.item(),
                "E_mix_w": Emix_w.item(),
                "normalization_penalty": normalization_penalty.item(),
                "alpha": self.rulsif_alpha,
                "ratio_real_mean": ratio_real.mean().item(),
                "ratio_fake_mean": ratio_fake.mean().item(),
            })

        elif self.loss_type == "kliep":
            # KLIEP (KL Importance Estimation Procedure)
            # Canonical form: w = exp(g - A), A = log E_r[exp(g)]
            # Loss: -E_q[g] + A (exact normalization)

            # Option 1: Exact canonical form (no stabilizers)
            if self.kliep_lambda == 0.0:
                # Canonical KLIEP: -E_q[g] + log E_r[exp(g)]
                g_real_clipped = torch.clamp(T_real, max=40.0)
                g_fake_clipped = torch.clamp(T_fake, max=40.0)

                Eq_g = g_real_clipped.mean()
                log_Er_exp_g = self.log_mean_exp(g_fake_clipped)

                loss = -Eq_g + log_Er_exp_g

                # Compute w for metrics
                with torch.no_grad():
                    w_fake = torch.exp(g_fake_clipped - log_Er_exp_g)
                    Er_w = w_fake.mean()

                metrics.update({
                    "kliep_loss": loss.detach().item(),
                    "E_q_g": Eq_g.item(),
                    "log_Er_exp_g": log_Er_exp_g.item(),
                    "E_r_w": Er_w.item(),
                })

            # Option 2: Penalized form with stabilizers
            else:
                g_real = self._apply_kliep_stabilizers(T_real)
                g_fake = self._apply_kliep_stabilizers(T_fake)

                # Compute w using chosen link function
                if self.use_exp_w:
                    Eq_logw = g_real.mean()
                    w_fake = torch.exp(torch.clamp(g_fake, max=40.0))
                else:
                    w_real = F.softplus(g_real) + EPS
                    w_fake = F.softplus(g_fake) + EPS
                    Eq_logw = torch.log(w_real.clamp(min=EPS)).mean()

                Er_w = w_fake.mean()

                # Penalized form: -E_q[log w] + λ(E_r[w] - 1)²
                loss = -Eq_logw + self.kliep_lambda * (Er_w - 1.0) ** 2

                metrics.update({
                    "kliep_loss": loss.detach().item(),
                    "E_q_logw": Eq_logw.item(),
                    "E_r_w": Er_w.item(),
                    "constraint_resid": (Er_w - 1.0).abs().item(),
                    "g_real_mean": g_real.mean().item(),
                    "g_fake_mean": g_fake.mean().item(),
                })

        return loss, metrics


if __name__ == "__main__":
    print("Testing density-ratio losses...")

    batch_size = 32

    # Test each loss type
    for loss_type in ["disc", "dv", "ulsif", "rulsif", "kliep", "infonce"]:
        print(f"\nTesting {loss_type}:")

        loss_fn = DensityRatioLoss(loss_type=loss_type)

        if loss_type == "infonce":
            # InfoNCE uses a (B, B) matrix
            T_mat = torch.randn(batch_size, batch_size)  # Score matrix T[i,j] = T(x_i, y_j)
            # Make diagonal larger to simulate positive pairs
            T_mat = T_mat + torch.eye(batch_size) * 2.0
            loss, metrics = loss_fn(T_mat)
        else:
            # Standard methods use 1-D tensors
            T_real = torch.randn(batch_size)  # Scores on real pairs
            T_fake = torch.randn(batch_size)  # Scores on fake pairs
            loss, metrics = loss_fn(T_real, T_fake)

        print(f"  Loss: {loss.item():.4f}")
        print(f"  Metrics: {list(metrics.keys())}")

    print("\nAll tests passed!")
