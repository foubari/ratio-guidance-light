"""
Training utilities for ratio estimator.
"""
import torch
from tqdm import tqdm


class RatioTrainer:
    """
    Trainer for density-ratio estimator.
    """
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        diffusion_schedule,
        device='cuda'
    ):
        """
        Args:
            model: RatioEstimator model
            loss_fn: DensityRatioLoss instance
            optimizer: Optimizer
            diffusion_schedule: DiffusionSchedule instance
            device: Device to train on
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.schedule = diffusion_schedule
        self.device = device

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: Dictionary with keys 'img1', 'img2', 'is_real'

        Returns:
            Dictionary with loss and metrics
        """
        img1 = batch['img1'].to(self.device)  # Standard MNIST [B, 1, 28, 28]
        img2 = batch['img2'].to(self.device)  # Rotated MNIST [B, 1, 28, 28]
        is_real = batch['is_real'].to(self.device)  # [B]

        batch_size = img1.shape[0]

        # Separate real and fake pairs
        real_mask = is_real > 0.5
        fake_mask = ~real_mask

        # Skip if we don't have both real and fake samples
        if real_mask.sum() == 0 or fake_mask.sum() == 0:
            return {'loss': 0.0, 'skipped': True}

        img1_real = img1[real_mask]
        img2_real = img2[real_mask]
        img1_fake = img1[fake_mask]
        img2_fake = img2[fake_mask]

        # Sample timesteps
        t_real = torch.randint(0, self.schedule.num_timesteps, (len(img1_real),), device=self.device)
        t_fake = torch.randint(0, self.schedule.num_timesteps, (len(img1_fake),), device=self.device)

        # Add noise to both images at the same timestep
        img1_real_noisy, _ = self.schedule.add_noise(img1_real, t_real)
        img2_real_noisy, _ = self.schedule.add_noise(img2_real, t_real)

        img1_fake_noisy, _ = self.schedule.add_noise(img1_fake, t_fake)
        img2_fake_noisy, _ = self.schedule.add_noise(img2_fake, t_fake)

        # Forward pass through ratio estimator
        scores_real = self.model(img1_real_noisy, img2_real_noisy, t_real)
        scores_fake = self.model(img1_fake_noisy, img2_fake_noisy, t_fake)

        # Compute loss
        loss, metrics = self.loss_fn(scores_real, scores_fake)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Add metadata to metrics
        metrics['loss'] = loss.item()
        metrics['t_mean'] = (t_real.float().mean().item() + t_fake.float().mean().item()) / 2
        metrics['num_real'] = len(img1_real)
        metrics['num_fake'] = len(img1_fake)

        return metrics

    def train_epoch(self, dataloader, epoch, total_epochs):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Dictionary with average metrics
        """
        self.model.train()
        epoch_metrics = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
        for batch in pbar:
            metrics = self.train_step(batch)

            if 'skipped' not in metrics:
                epoch_metrics.append(metrics)

                # Update progress bar
                if len(epoch_metrics) % 10 == 0:
                    avg_loss = sum(m['loss'] for m in epoch_metrics[-10:]) / min(10, len(epoch_metrics))
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Compute average metrics
        if len(epoch_metrics) == 0:
            return {'loss': float('inf')}

        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            if key != 'skipped':
                avg_metrics[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)

        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader):
        """
        Validate on a dataset.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Dictionary with average metrics
        """
        self.model.eval()
        val_metrics = []

        for batch in dataloader:
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            is_real = batch['is_real'].to(self.device)

            real_mask = is_real > 0.5
            fake_mask = ~real_mask

            if real_mask.sum() == 0 or fake_mask.sum() == 0:
                continue

            img1_real = img1[real_mask]
            img2_real = img2[real_mask]
            img1_fake = img1[fake_mask]
            img2_fake = img2[fake_mask]

            # Sample timesteps
            t_real = torch.randint(0, self.schedule.num_timesteps, (len(img1_real),), device=self.device)
            t_fake = torch.randint(0, self.schedule.num_timesteps, (len(img1_fake),), device=self.device)

            # Add noise
            img1_real_noisy, _ = self.schedule.add_noise(img1_real, t_real)
            img2_real_noisy, _ = self.schedule.add_noise(img2_real, t_real)
            img1_fake_noisy, _ = self.schedule.add_noise(img1_fake, t_fake)
            img2_fake_noisy, _ = self.schedule.add_noise(img2_fake, t_fake)

            # Forward pass
            scores_real = self.model(img1_real_noisy, img2_real_noisy, t_real)
            scores_fake = self.model(img1_fake_noisy, img2_fake_noisy, t_fake)

            # Compute loss
            loss, metrics = self.loss_fn(scores_real, scores_fake)
            metrics['loss'] = loss.item()

            val_metrics.append(metrics)

        # Compute average metrics
        if len(val_metrics) == 0:
            return {'loss': float('inf')}

        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in val_metrics) / len(val_metrics)

        return avg_metrics


if __name__ == "__main__":
    print("Testing RatioTrainer...")

    from models.ratio_estimator import RatioEstimator
    from utils.losses import DensityRatioLoss
    from utils.diffusion import DiffusionSchedule
    from data.mnist_dataset import get_ratio_dataloader

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = RatioEstimator().to(device)
    loss_fn = DensityRatioLoss(loss_type='disc')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schedule = DiffusionSchedule(num_timesteps=1000, device=device)

    trainer = RatioTrainer(model, loss_fn, optimizer, schedule, device)

    # Get dataloader
    dataloader = get_ratio_dataloader(batch_size=32, train=True, num_workers=0)

    # Test one training step
    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)
    print(f"✓ Training step completed: loss={metrics.get('loss', 'N/A')}")

    print("RatioTrainer test passed!")
