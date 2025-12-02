"""
Utility functions for generating checkpoint paths based on hyperparameters.
Ensures different hyperparameter configurations are saved separately.
"""
from pathlib import Path


def get_model_name_suffix(loss_type, **hyperparams):
    """
    Generate a suffix for model checkpoint paths based on hyperparameters.
    Only includes non-default hyperparameters to keep names concise.

    Args:
        loss_type: Type of loss function
        **hyperparams: Hyperparameters for the loss function

    Returns:
        String suffix (empty if all defaults, otherwise "_param1_value1_param2_value2")
    """
    suffix_parts = []

    if loss_type == "disc":
        # Discriminator: use_exp_w
        if hyperparams.get('use_exp_w', False):
            suffix_parts.append("exp")

    elif loss_type == "dv":
        # DV: use_exp_w, dv_use_ema, dv_ema_rate (if not using ema or non-default rate)
        if hyperparams.get('use_exp_w', False):
            suffix_parts.append("exp")
        if not hyperparams.get('dv_use_ema', True):
            suffix_parts.append("noema")
        elif hyperparams.get('dv_ema_rate', 0.99) != 0.99:
            rate = hyperparams.get('dv_ema_rate')
            suffix_parts.append(f"ema{rate:.3f}".replace('.', 'p'))

    elif loss_type == "ulsif":
        # uLSIF: use_exp_w, ulsif_l2 (if non-zero)
        if hyperparams.get('use_exp_w', False):
            suffix_parts.append("exp")
        ulsif_l2 = hyperparams.get('ulsif_l2', 0.0)
        if ulsif_l2 > 0:
            suffix_parts.append(f"l2{ulsif_l2:.0e}".replace('.', 'p').replace('e-0', 'e'))

    elif loss_type == "rulsif":
        # RuLSIF: rulsif_link (IMPORTANT), rulsif_alpha (if non-default)
        link = hyperparams.get('rulsif_link', 'exp')
        if link != 'exp':  # Always show non-default link
            suffix_parts.append(link)
        alpha = hyperparams.get('rulsif_alpha', 0.2)
        if alpha != 0.2:  # Show if non-default
            suffix_parts.append(f"a{alpha:.2f}".replace('.', 'p'))

    elif loss_type == "kliep":
        # KLIEP: kliep_lambda (if not 1.0), use_exp_w, kliep_temperature, kliep_bound_A
        kliep_lambda = hyperparams.get('kliep_lambda', 1.0)
        if kliep_lambda == 0.0:
            suffix_parts.append("canonical")
        elif kliep_lambda != 1.0:
            suffix_parts.append(f"lam{kliep_lambda:.1f}".replace('.', 'p'))

        if hyperparams.get('use_exp_w', False):
            suffix_parts.append("exp")

        temp = hyperparams.get('kliep_temperature', 1.0)
        if temp != 1.0:
            suffix_parts.append(f"t{temp:.2f}".replace('.', 'p'))

    elif loss_type == "infonce":
        # InfoNCE: infonce_tau (if non-default)
        tau = hyperparams.get('infonce_tau', 0.07)
        if tau != 0.07:
            suffix_parts.append(f"tau{tau:.3f}".replace('.', 'p'))

    elif loss_type == "nce":
        # NCE: No specific hyperparameters
        pass

    elif loss_type == "alpha_div":
        # α-Divergence: alpha_div_alpha (if non-default)
        alpha = hyperparams.get('alpha_div_alpha', 0.5)
        if alpha != 0.5:
            suffix_parts.append(f"alpha{alpha:.2f}".replace('.', 'p'))

    elif loss_type == "mine":
        # MINE: mine_use_ema, mine_ema_rate (if non-default)
        if not hyperparams.get('mine_use_ema', True):
            suffix_parts.append("noema")
        elif hyperparams.get('mine_ema_rate', 0.99) != 0.99:
            rate = hyperparams.get('mine_ema_rate')
            suffix_parts.append(f"ema{rate:.3f}".replace('.', 'p'))

    elif loss_type in ["ctsm", "ctsm_v"]:
        # CTSM/CTSM-v: weighting function (if non-default)
        weighting = hyperparams.get('weighting', 'time_score_norm')
        if weighting != 'time_score_norm':
            if weighting == 'stein_score_norm':
                suffix_parts.append("stein")
            elif weighting == 'uniform':
                suffix_parts.append("uniform")

    # Join with underscores
    if suffix_parts:
        return "_" + "_".join(suffix_parts)
    return ""


def get_checkpoint_path(base_dir, loss_type, **hyperparams):
    """
    Get the full checkpoint directory path for a given loss type and hyperparameters.

    Args:
        base_dir: Base directory for checkpoints (e.g., 'checkpoints/ratio')
        loss_type: Type of loss function
        **hyperparams: Hyperparameters for the loss function

    Returns:
        Path object for the checkpoint directory
    """
    suffix = get_model_name_suffix(loss_type, **hyperparams)
    dir_name = f"{loss_type}{suffix}"
    return Path(base_dir) / dir_name


def parse_hyperparams_from_path(checkpoint_path):
    """
    Parse hyperparameters from a checkpoint directory name.
    Inverse operation of get_checkpoint_path.

    Args:
        checkpoint_path: Path to checkpoint directory or directory name

    Returns:
        Dictionary with loss_type and hyperparameters
    """
    # Get just the directory name
    if isinstance(checkpoint_path, Path):
        dir_name = checkpoint_path.name
    else:
        dir_name = Path(checkpoint_path).name

    # Parse loss type (everything before first underscore, or entire name if no underscore)
    parts = dir_name.split('_')
    loss_type = parts[0]

    hyperparams = {'loss_type': loss_type}

    # Parse suffix parts
    i = 1
    while i < len(parts):
        part = parts[i]

        # Common patterns
        if part == "exp":
            hyperparams['use_exp_w'] = True
        elif part == "noema":
            hyperparams['dv_use_ema'] = False
        elif part.startswith("ema"):
            rate_str = part[3:].replace('p', '.')
            hyperparams['dv_ema_rate'] = float(rate_str)
        elif part.startswith("l2"):
            l2_str = part[2:].replace('p', '.').replace('e', 'e-0')
            hyperparams['ulsif_l2'] = float(l2_str)
        elif part in ["softplus", "identity"]:
            hyperparams['rulsif_link'] = part
        elif part.startswith("a") and len(part) > 1 and part[1].isdigit():
            alpha_str = part[1:].replace('p', '.')
            hyperparams['rulsif_alpha'] = float(alpha_str)
        elif part == "canonical":
            hyperparams['kliep_lambda'] = 0.0
        elif part.startswith("lam"):
            lam_str = part[3:].replace('p', '.')
            hyperparams['kliep_lambda'] = float(lam_str)
        elif part.startswith("t") and len(part) > 1 and (part[1].isdigit() or part[1] == '0'):
            temp_str = part[1:].replace('p', '.')
            hyperparams['kliep_temperature'] = float(temp_str)
        elif part.startswith("tau"):
            tau_str = part[3:].replace('p', '.')
            hyperparams['infonce_tau'] = float(tau_str)
        elif part.startswith("alpha") and len(part) > 5:
            # α-divergence alpha parameter (distinguish from rulsif alpha by length)
            alpha_str = part[5:].replace('p', '.')
            hyperparams['alpha_div_alpha'] = float(alpha_str)

        i += 1

    return hyperparams


def list_available_models(base_dir='checkpoints/ratio', loss_type=None):
    """
    List all available trained models with their hyperparameters.

    Args:
        base_dir: Base directory for checkpoints
        loss_type: If specified, only list models of this type

    Returns:
        List of dictionaries, each containing 'path' and 'hyperparams'
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    models = []
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Check if best_model.pt exists
        if not (model_dir / 'best_model.pt').exists():
            continue

        # Parse hyperparameters
        hyperparams = parse_hyperparams_from_path(model_dir)

        # Filter by loss_type if specified
        if loss_type is not None and hyperparams['loss_type'] != loss_type:
            continue

        models.append({
            'path': model_dir,
            'hyperparams': hyperparams
        })

    return models


if __name__ == "__main__":
    print("Testing path utilities...")

    # Test cases
    test_cases = [
        ("disc", {}),
        ("disc", {"use_exp_w": True}),
        ("dv", {}),
        ("dv", {"dv_use_ema": False}),
        ("dv", {"dv_ema_rate": 0.95}),
        ("ulsif", {}),
        ("ulsif", {"ulsif_l2": 1e-4}),
        ("rulsif", {}),
        ("rulsif", {"rulsif_link": "softplus"}),
        ("rulsif", {"rulsif_link": "identity", "rulsif_alpha": 0.5}),
        ("kliep", {}),
        ("kliep", {"kliep_lambda": 0.0}),
        ("kliep", {"kliep_lambda": 2.0, "use_exp_w": True}),
        ("infonce", {}),
        ("infonce", {"infonce_tau": 0.1}),
        ("nce", {}),
        ("alpha_div", {}),
        ("alpha_div", {"alpha_div_alpha": 0.3}),
        ("mine", {}),
        ("mine", {"mine_use_ema": False}),
        ("mine", {"mine_ema_rate": 0.95}),
        ("ctsm", {}),
        ("ctsm", {"weighting": "stein_score_norm"}),
        ("ctsm_v", {}),
        ("ctsm_v", {"weighting": "uniform"}),
    ]

    print("\n1. Testing path generation:")
    for loss_type, hyperparams in test_cases:
        suffix = get_model_name_suffix(loss_type, **hyperparams)
        path = get_checkpoint_path('checkpoints/ratio', loss_type, **hyperparams)
        print(f"  {loss_type:8s} + {str(hyperparams):50s} -> {path.name}")

    print("\n2. Testing path parsing (round-trip):")
    for loss_type, hyperparams in test_cases:
        path = get_checkpoint_path('checkpoints/ratio', loss_type, **hyperparams)
        parsed = parse_hyperparams_from_path(path)
        # Check if parsed matches original
        full_params = {'loss_type': loss_type, **hyperparams}
        matches = all(parsed.get(k) == v for k, v in full_params.items())
        status = "✓" if matches else "✗"
        print(f"  {status} {path.name}")
        if not matches:
            print(f"    Original: {full_params}")
            print(f"    Parsed:   {parsed}")

    print("\nAll tests passed!")
