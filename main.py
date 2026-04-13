import argparse
import os
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from tqdm import tqdm

from model import BoaConstrictor, ByteDataloader, make_splits
from boa import BOA
from train import train


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_arg: str, experiments_root: Path = Path('experiments')) -> Path:
    """Resolve a --config argument which may be a path or an experiment name.

    Order:
      1. If the argument is an existing file path, return it.
      2. If it's a simple experiment name (no existing path), look for
         experiments/<name>/<name>.yaml and return if exists.
      3. Fallback to configs/<name>.yaml if present.
      4. Raise FileNotFoundError.
    """
    if config_arg is None:
        return None
    p = Path(config_arg)
    # Direct file path provided
    if p.exists():
        return p

    # Try experiments/<name>/<name>.yaml
    name = p.stem
    exp_cfg = experiments_root / name / f"{name}.yaml"
    if exp_cfg.exists():
        return exp_cfg

    # Try configs/<name>.yaml
    cfg_cfg = Path('configs') / f"{name}.yaml"
    if cfg_cfg.exists():
        return cfg_cfg

    raise FileNotFoundError(f"Could not resolve config argument '{config_arg}' to a config file")


def parse_args():
    p = argparse.ArgumentParser(description="Run BoaConstrictor experiments from a config file")
    p.add_argument('--config', '-c', type=Path, required=False, help='Path to YAML experiment config')
    p.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    p.add_argument('--device', type=str, default="cuda", help='Torch device override (cpu|cuda)')
    p.add_argument('--precision', type=str, default=None, choices=['fp32','bf16','fp16', 'fp8'], help='Precision override')
    p.add_argument('--compile', action='store_true', help='Use torch.compile for faster training (adds ~60s warmup)')
    p.add_argument('--new-experiment', action='store_true', help='Create a new experiment config interactively and run it')
    p.add_argument('--train-only', action='store_true', help='Only run training')
    p.add_argument('--compress-only', action='store_true', help='Only run compression')
    p.add_argument('--decompress-only', action='store_true', help='Only run decompression')
    p.add_argument('--show-timings', action='store_true', help='Print timings for each major operation')
    p.add_argument('--verify', action='store_true', help='After decompression, verify bytes match the input file used for compression')
    p.add_argument('--evaluate', action='store_true', help='After decompression, run evaluation metrics on the compressor model')
    p.add_argument('--evaluate-only', action='store_true', help='After decompression, run evaluation metrics on the compressor model')
    p.add_argument('--comparison-baseline-only', action='store_true', help='Run LZMA, ZLIB, ZSTD, LZ4 baseline compressions on the compression input file, print results, and exit')
    p.add_argument('--model-path', type=str, default=None, help='Path to a pre-trained model .pt file (state_dict or full model). If provided, training is skipped and the model is loaded')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (sets torch, numpy, python random seeds)')
    p.add_argument('--backbone', type=str, default="mamba",
                   choices=['mamba', 'mambav1', 'mamba2', 'lstm', 'gru', 'mingru',
                            'transformer', 'rwkv6', 'rwkv7', 'rwkv8', 'griffin', 'xlstm'],
                   help='Model backbone override (default: mamba)')
    # --- Ablation overrides ---
    p.add_argument('--d-model', type=int, default=None, help='Override d_model from config')
    p.add_argument('--num-layers', type=int, default=None, help='Override num_layers from config')
    p.add_argument('--data-frac', type=float, default=1.0, help='Fraction of training data to use (0.0-1.0, default: 1.0)')
    p.add_argument('--adapt', type=float, nargs=2, metavar=('LR', 'STEPS'),
                   help='Enable online head adaptation with given learning rate and number of steps (e.g. --adapt 1e-3 1)')
    p.add_argument('--measure-energy', action='store_true', help='Measure energy consumption during compression/decompression using CPPJoules (RAPL + NVML)')
    p.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    p.add_argument('--lr', type=float, default=None, help='Override learning rate')
    p.add_argument('--patience', type=int, default=None, help='Override early stopping patience (0=disabled)')
    p.add_argument('--force-train', action='store_true', help='Force training from scratch, ignoring existing checkpoints')
    # --- Weights & Biases ---
    p.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', type=str, default='boa-constrictor', help='W&B project name')
    p.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (team or username)')
    p.add_argument('--wandb-name', type=str, default=None, help='W&B run name (defaults to experiment name)')
    p.add_argument('--wandb-tags', type=str, nargs='*', default=None, help='W&B run tags')
    return p.parse_args()


def main():
    args = parse_args()

    # Set random seeds for reproducibility [Referee Major #9]
    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[INFO] Random seed set to {args.seed}")

    # If user requests a new experiment, run interactive creator and obtain a config path
    if args.new_experiment:
        def _prompt(prompt, default=None, cast=str):
            if default is None:
                resp = input(f"{prompt}: ").strip()
            else:
                resp = input(f"{prompt} [{default}]: ").strip()
                if resp == "":
                    resp = str(default)
            try:
                return cast(resp)
            except Exception:
                return resp

        print("Creating a new experiment config interactively. Press enter to accept the default shown in brackets.")
        name = _prompt("Experiment name", "example_experiment")
        file_path = _prompt("Path to dataset file (binary)", "/path/to/dataset.bin")
        progress = _prompt("Show progress bars (true/false)", "true", lambda s: s.lower() in ("1","true","yes"))
        device = _prompt("Device (cpu|cuda)", "cuda")
        precision = _prompt("Precision (fp32|fp16|fp8)", "fp32")
        seq_len = _prompt("Sequence length (seq_len)", 32768, int)
        batch_size = _prompt("Batch size", 3, int)
        d_model = _prompt("Model d_model", 256, int)
        num_layers = _prompt("Model num_layers", 2, int)
        lr = _prompt("Learning rate", 5e-4, float)
        epochs = _prompt("Epochs", 10, int)
        chunks_count = _prompt("Compression chunks_count", 1000, int)
        use_vocab_subset = _prompt("Use vocab subset (true/false)", "false", lambda s: s.lower() in ("1","true","yes"))
        compress_file = _prompt("File to compress (leave blank to use dataset file)", "", lambda s: s if s != "" else "")
        backbone_choice = _prompt("Backbone (mamba|mambav1|mamba2|lstm|gru|mingru|transformer|rwkv6|rwkv7|rwkv8|griffin|xlstm)", "mamba")
        splits_in = _prompt("Data splits as comma-separated (train,val,test)", "0.8,0.1,0.1")
        try:
            splits = [float(x.strip()) for x in splits_in.split(',')]
            if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
                print("Warning: splits do not sum to 1. Using default [0.8,0.1,0.1].")
                splits = [0.8, 0.1, 0.1]
        except Exception:
            splits = [0.8, 0.1, 0.1]

        cfg = {
            'name': name,
            'file_path': file_path,
            'progress': bool(progress),
            'device': device,
            'precision': precision,
            'dataloader': {'seq_len': int(seq_len), 'batch_size': int(batch_size)},
            'model': {'d_model': int(d_model), 'num_layers': int(num_layers), 'backbone': backbone_choice},
            'training': {'lr': float(lr), 'epochs': int(epochs)},
            'compression': {'chunks_count': int(chunks_count), 'file_to_compress': compress_file},
            'use_vocab_subset': bool(use_vocab_subset),
            'splits': splits
        }

        # Decide where to save the config: store it under experiments/<name>/<name>.yaml
        cfg_path = Path('experiments') / name / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        print(f"Wrote new experiment config to: {cfg_path}")

        # Use the newly created config for the rest of the run
        args.config = str(cfg_path)

    if args.config is None:
        raise ValueError('Either --config must be provided or use --new-experiment to create one interactively')

    # Resolve the config argument: allow passing an experiment name which maps
    # to experiments/<name>/<name>.yaml, or a direct path.
    args.config = resolve_config_path(str(args.config))
    config = load_config(args.config)

    # Apply CLI overrides
    progress = not args.no_progress and config.get('progress', True)
    device =  config.get('device', 'cuda' if torch.cuda.is_available() else 'cuda') or args.device
    
    print(device)
    precision = args.precision or config.get('precision', 'fp32')
    verify = args.verify or bool(config.get('verify', False))
    # Model path can be provided via CLI or config (either top-level 'model_path' or under 'model.path')
    model_path_cfg = config.get('model_path') or config.get('model', {}).get('path')
    model_path = Path(args.model_path).expanduser() if args.model_path else (Path(model_path_cfg).expanduser() if model_path_cfg else None)
    if model_path is not None:
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not model_path.is_absolute():
            model_path = (cfg_dir / model_path).resolve()

    # Experiment parameters (with sensible defaults)
    # Use the config filename stem as the canonical experiment/model name
    # so checkpoints are consistently named and retraining can be skipped.
    name = Path(args.config).stem
    file_path = config.get('file_path', '')
    # Resolve file_path: if it's absolute, use as-is; if relative, interpret
    # it relative to the directory of the resolved config file (so passing
    # --config <experiment_name> works and paths inside the YAML are relative
    # to that YAML file).
    if file_path:
        file_path = Path(file_path)
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not file_path.is_absolute():
            file_path = (cfg_dir / file_path).resolve()
    seq_len = config.get('dataloader', {}).get('seq_len', 32768)
    batch_size = config.get('dataloader', {}).get('batch_size', 3)
    d_model = args.d_model if args.d_model is not None else config.get('model', {}).get('d_model', 256)
    num_layers = args.num_layers if args.num_layers is not None else config.get('model', {}).get('num_layers', 8)
    # Backbone selection (default: mamba for backward compatibility)
    model_cfg = config.get('model', {})
    backbone = model_cfg.get('backbone', 'mamba') or args.backbone
    _MODEL_KNOWN_KEYS = {'d_model', 'num_layers', 'backbone', 'path'}
    backbone_kwargs = {k: v for k, v in model_cfg.items() if k not in _MODEL_KNOWN_KEYS}
    # Log transformer-specific defaults that differ from the old version
    if backbone == 'transformer':
        backbone_kwargs.setdefault('window_size', 4096)
        backbone_kwargs.setdefault('n_sink', 4)
    elif backbone == 'rwkv6':
        backbone_kwargs.setdefault('n_heads', 4)
    elif backbone == 'rwkv7':
        backbone_kwargs.setdefault('n_heads', 4)
    elif backbone == 'griffin':
        backbone_kwargs.setdefault('n_heads', 4)
        backbone_kwargs.setdefault('local_window', 128)
    elif backbone == 'xlstm':
        backbone_kwargs.setdefault('n_heads', 4)
    lr = args.lr if args.lr is not None else float(config.get('training', {}).get('lr', 5e-4))
    num_epochs = args.epochs if args.epochs is not None else config.get('training', {}).get('epochs', 50)
    use_vocab_subset = config.get('use_vocab_subset', False)

    # Build a unique model name when CLI overrides are active so that each
    # ablation configuration gets its own checkpoint files instead of
    # overwriting the single default checkpoint.
    _name_parts = []
    if args.d_model is not None:
        _name_parts.append(f"d{d_model}")
    if args.num_layers is not None:
        _name_parts.append(f"L{num_layers}")
    data_frac = getattr(args, 'data_frac', 1.0)
    if data_frac < 1.0:
        _name_parts.append(f"f{data_frac:.2f}")
    if args.seed is not None and args.seed != 42:
        _name_parts.append(f"s{args.seed}")
    if _name_parts:
        name = f"{name}_{'_'.join(_name_parts)}"
        print(f"[INFO] Model name adjusted for ablation: {name}")

    timings = {}

    # Read file
    t0 = time.perf_counter()
    if not file_path:
        raise ValueError('file_path must be set in the config or passed via CLI')

    # file_path is already a Path (resolved above when possible)
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data_bytes = f.read()

    timings['read_bytes'] = time.perf_counter() - t0
    print(f"Read {len(data_bytes)} bytes from {file_path} in {timings['read_bytes']:.2f}s")
    compress_file_cfg = config.get('compression', {}).get('file_to_compress', '')
    # If blank, use the original dataset file we already loaded
    if not compress_file_cfg:
        compress_file_path = file_path
    else:
        # Resolve compress_file relative to config dir when relative
        cfp = Path(compress_file_cfg)
        cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp

    # Compute vocabulary and remap (training data only)
    if use_vocab_subset:
        unique_bytes = sorted(list(set(data_bytes)))
        vocab_size = len(unique_bytes)
        print(f"Using vocab subset of size {vocab_size} out of 256 possible bytes.")
        byte_to_idx = {b: i for i, b in enumerate(unique_bytes)}
        idx_to_byte = {i: b for i, b in enumerate(unique_bytes)}

        # Remap training data
        arr = np.frombuffer(data_bytes, dtype=np.uint8)
        lookup = np.zeros(256, dtype=np.uint8)
        for b, idx in byte_to_idx.items():
            lookup[b] = idx
        data_bytes = lookup[arr].tobytes()
    else:
        vocab_size = 256
        unique_bytes = None
        byte_to_idx = None
        idx_to_byte = None
        lookup = None

    # Prepare experiment output directory and filenames (needed before optional training)
    experiments_root = Path(config.get('experiments_root', 'experiments'))
    exp_dir = experiments_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup model, dataloaders, optimizer, loss
    print(f"[INFO] Backbone: {backbone}" + (f"  kwargs: {backbone_kwargs}" if backbone_kwargs else ""))
    model = BoaConstrictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size,
                           device=device, backbone=backbone, **backbone_kwargs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    dataloader = ByteDataloader(data_bytes, seq_len=seq_len, batch_size=batch_size, device=device)

    train_b, val_b, test_b = make_splits(data_bytes, dataloader.seq_len, dataloader.batch_size,
                                         splits=tuple(config.get('splits', (0.8, 0.1, 0.1))))

    # --- Data-fraction subsampling (for ablation studies) ---
    data_frac = getattr(args, 'data_frac', 1.0)
    if data_frac < 1.0:
        n_keep = int(len(train_b) * data_frac)
        block = dataloader.seq_len * dataloader.batch_size
        n_keep = (n_keep // block) * block  # align to batch boundaries
        n_keep = max(n_keep, block)  # keep at least one batch
        train_b = train_b[:n_keep]
        print(f"[INFO] Data fraction: {data_frac:.0%} — using {len(train_b):,} / {len(train_b)//data_frac:,.0f} training bytes")

    train_loader = ByteDataloader(train_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    val_loader = ByteDataloader(val_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    test_loader = ByteDataloader(test_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = float(config.get('training', {}).get('weight_decay', 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # If a model path is provided and exists, load it and skip training
    def _infer_backbone_from_keys(keys):
        """Infer backbone type from state_dict key patterns.

        Each backbone produces distinctive key names inside
        ``blocks.<i>.`` — we check for those fingerprints.
        """
        ks = set(keys)
        has = lambda pat: any(pat in k for k in ks)  # noqa: E731

        # Unique sub-module names per backbone
        # RWKV-8 ROSA has .rosa_mix.rosa.
        if has('.rosa_mix.'):  return 'rwkv8'
        # RWKV-7 has .time_mix.W_a (non-diagonal transition); check before rwkv6
        if has('.time_mix.') and has('.time_mix.W_a.'): return 'rwkv7'
        if has('.time_mix.'):   return 'rwkv6'
        if has('.rg_lru.'):    return 'griffin'
        if has('.mlstm.'):     return 'xlstm'
        if has('.mamba2.'):    return 'mamba2'
        if has('.mingru.'):    return 'mingru'
        if has('.lstm.'):      return 'lstm'
        if has('.gru.'):       return 'gru'
        # Transformer has q_proj but no recurrence sub-modules
        if has('.q_proj.') or has('.kv_proj.'):
            return 'transformer'
        # Distinguish mamba (improved) vs mambav1 by FFN structure
        if has('.mamba.'):
            # Improved mamba uses SwiGLU (ff.w_gate) + final_norm
            if has('.ff.w_gate.') and has('final_norm.'):
                return 'mamba'
            return 'mambav1'
        return None

    def _load_model_from_path(model, path: Path, expected_backbone: str = None):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = torch.load(path, map_location='cpu')

        saved_backbone = None
        state_dict = None

        if isinstance(obj, dict):
            # New backbone-aware format: {'backbone': str, 'state_dict': dict}
            if 'backbone' in obj and 'state_dict' in obj:
                saved_backbone = obj['backbone']
                state_dict = obj['state_dict']
            # Legacy format with state_dict under a key
            elif 'state_dict' in obj:
                state_dict = obj['state_dict']
            # Pure state_dict (all string keys, no 'backbone' key)
            elif all(isinstance(k, str) for k in obj.keys()):
                state_dict = obj

        if state_dict is not None:
            # For legacy checkpoints, infer backbone from key patterns
            if saved_backbone is None:
                inferred = _infer_backbone_from_keys(state_dict.keys())
                if inferred:
                    saved_backbone = inferred
                    print(f"  Inferred backbone from checkpoint keys: {inferred}")

            # Backbone compatibility check
            if saved_backbone and expected_backbone and saved_backbone != expected_backbone:
                raise ValueError(
                    f"Backbone mismatch: checkpoint was trained with "
                    f"'{saved_backbone}' but current config specifies "
                    f"'{expected_backbone}'. Change backbone in config to "
                    f"'{saved_backbone}' or retrain."
                )
            if saved_backbone:
                print(f"  Checkpoint backbone: {saved_backbone}")
            elif expected_backbone:
                print(f"  Legacy checkpoint (could not infer backbone). "
                      f"Assuming compatible with '{expected_backbone}'.")

            # Load with strict=True first to catch mismatches cleanly
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"  [WARN] state_dict load: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected keys")
                if len(missing) > len(state_dict) * 0.5:
                    raise ValueError(
                        f"Too many missing keys ({len(missing)}/{len(state_dict)}). "
                        f"This checkpoint is likely incompatible with backbone "
                        f"'{expected_backbone}'."
                    )
            return model

        # If whole model was saved
        if hasattr(obj, 'state_dict') and hasattr(obj, 'parameters'):
            if expected_backbone and hasattr(obj, '_backbone_name'):
                if obj._backbone_name != expected_backbone:
                    raise ValueError(
                        f"Backbone mismatch: saved model uses "
                        f"'{obj._backbone_name}' but config specifies "
                        f"'{expected_backbone}'."
                    )
            return obj

        raise ValueError(f"Unrecognized checkpoint format at {path}")

    # If no explicit model_path was provided, check for an existing final checkpoint
    start_epoch = 1
    resume_training = False
    force_train = getattr(args, 'force_train', False)
    
    default_ckpt = exp_dir / f"{name}_final_model_{precision}.pt"
    if force_train:
        model_path = None  # ignore any existing checkpoint
        print("[INFO] --force-train: training from scratch (ignoring existing checkpoints)")
    elif model_path is None:
        if default_ckpt.exists():
            model_path = default_ckpt
            print(f"Found existing checkpoint at {model_path}. Will load and skip training.")
        else:
            # Check for intermediate checkpoints
            candidates = list(exp_dir.glob(f"{name}_*_Checkpoint_epoch_*_{precision}.pt"))
            if candidates:
                def _get_epoch_from_path(p: Path) -> int:
                    parts = p.stem.split('_')
                    if 'epoch' in parts:
                        try:
                            return int(parts[parts.index('epoch') + 1])
                        except (ValueError, IndexError):
                            pass
                    return 0

                latest_ckpt = max(candidates, key=_get_epoch_from_path)
                latest_epoch = _get_epoch_from_path(latest_ckpt)
                
                if latest_epoch > 0:
                    model_path = latest_ckpt
                    if latest_epoch < num_epochs:
                        start_epoch = latest_epoch + 1
                        resume_training = True
                        print(f"Found intermediate checkpoint at {model_path}. Resuming training from epoch {start_epoch}.")
                    else:
                        print(f"Found checkpoint at {model_path} (epoch {latest_epoch} >= {num_epochs}). Will load and skip training.")

    # Training or loading
    if model_path is not None and Path(model_path).exists() and not args.comparison_baseline_only:
        print(f"Loading pre-trained model from {model_path}")
        if not resume_training:
            print("Skipping training (final model or explicit path provided).")
        
        t_start = time.perf_counter()
        model = _load_model_from_path(model, Path(model_path),
                                       expected_backbone=backbone)
        model = model.to(device)
        timings['load_model'] = time.perf_counter() - t_start
        print(f"Model loaded in {timings['load_model']:.2f}s")

    if not args.compress_only and not args.decompress_only and not args.comparison_baseline_only:
        if model_path is None or resume_training:
            print(f"Starting training on device=={device}, precision={precision}, backbone={backbone}, epochs={num_epochs}, start_epoch={start_epoch}")
            t_start = time.perf_counter()
            grad_clip = float(config.get('training', {}).get('grad_clip', 1.0))

            # LR scheduler (optional, from config)
            scheduler_name = config.get('training', {}).get('scheduler', None)
            scheduler = None
            if scheduler_name == 'cosine':
                remaining_epochs = max(1, num_epochs - start_epoch + 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

            patience = args.patience if args.patience is not None else int(config.get('training', {}).get('patience', 0))

            # --- Weights & Biases initialisation ---
            wandb_run = None
            if args.wandb:
                try:
                    import wandb
                    wandb_run = wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=args.wandb_name or name,
                        tags=args.wandb_tags,
                        config={
                            "experiment": name,
                            "backbone": backbone,
                            "d_model": d_model,
                            "num_layers": num_layers,
                            "seq_len": seq_len,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "epochs": num_epochs,
                            "start_epoch": start_epoch,
                            "precision": precision,
                            "vocab_size": vocab_size,
                            "patience": patience,
                            "grad_clip": grad_clip,
                            "scheduler": scheduler_name,
                            "num_params": num_params,
                            "device": device,
                            "dataset": str(file_path),
                            "seed": args.seed,
                            "data_frac": getattr(args, 'data_frac', 1.0),
                            "train_bytes": len(train_b),
                            **backbone_kwargs,
                        },
                    )
                    wandb.watch(model, log="gradients", log_freq=100)
                    print(f"[INFO] W&B run: {wandb_run.url}")
                except ImportError:
                    print("[WARN] --wandb requested but `wandb` is not installed. Continuing without it.")
                except Exception as e:
                    print(f"[WARN] W&B init failed: {e}. Continuing without it.")

            # Optional torch.compile for faster training
            if getattr(args, 'compile', False):
                print("[INFO] Compiling model with torch.compile (mode='max-autotune')...")
                model = torch.compile(model, mode='max-autotune')

            train(model, train_loader, val_loader, test_loader, optimizer, criterion,
                  device=device, name=str(exp_dir / name), NUM_EPOCHS=num_epochs, PRECISION=precision, progress=progress, start_epoch=start_epoch, vocab_size=vocab_size, grad_clip=grad_clip, scheduler=scheduler, patience=patience, wandb_run=wandb_run)

            # Finish wandb run after training
            if wandb_run is not None:
                wandb_run.finish()

            timings['training'] = time.perf_counter() - t_start
            print(f"Training complete in {timings['training']:.2f}s")

            # After successful training, persist model_path into the YAML config
            trained_ckpt = exp_dir / f"{name}_final_model_{precision}.pt"
            final_model_name = f"{name}_final_model_{precision}.pt"
            try:
                cfg_path: Path = Path(args.config)
                # Write model_path relative to the config directory when possible
                rel_ckpt = trained_ckpt
                if trained_ckpt.is_absolute():
                    try:
                        rel_ckpt = trained_ckpt.relative_to(cfg_path.parent)
                    except Exception:
                        rel_ckpt = trained_ckpt
                with open(cfg_path, 'r') as f:
                    cfg_data = yaml.safe_load(f) or {}
                cfg_data['model_path'] = str(final_model_name)
                with open(cfg_path, 'w') as f:
                    yaml.safe_dump(cfg_data, f)
                print(f"Updated config with model_path: {rel_ckpt}")
            except Exception as e:
                print(f"[WARN] Failed to update config with model_path: {e}")

    compress_file_cfg = config.get('compression', {}).get('file_to_compress', '')
    # If blank, use the original dataset file we already loaded
    if not compress_file_cfg:
        compress_file_path = file_path
    else:
        # Resolve compress_file relative to config dir when relative
        cfp = Path(compress_file_cfg)
        cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp

    # If the user only wants baseline comparisons, run quick LZMA and ZLIB (ultra) compressions
    # on the compression input file and print results, then exit.
    def _run_baseline_comparisons(in_path: Path, out_dir: Path, exp_name: str):
        import lzma
        import zlib
        import bz2
        import uproot
        import time

        # Import ZSTD and LZ4 for broader baseline coverage [Referee Major #2]
        try:
            import zstandard as zstd
            _has_zstd = True
        except ImportError:
            _has_zstd = False
            print("[WARN] zstandard not installed; skipping ZSTD baselines (pip install zstandard)")
        try:
            import lz4.frame as lz4f
            import lz4.block as lz4b
            _has_lz4 = True
        except ImportError:
            _has_lz4 = False
            print("[WARN] lz4 not installed; skipping LZ4 baselines (pip install lz4)")
        try:
            import brotli
            _has_brotli = True
        except ImportError:
            _has_brotli = False
            print("[WARN] brotli not installed; skipping Brotli baselines (pip install brotli)")
        try:
            import pyppmd
            _has_ppmd = True
        except ImportError:
            _has_ppmd = False
            print("[WARN] pyppmd not installed; skipping PPMd baselines (pip install pyppmd)")
        try:
            import constriction
            _has_constriction = True
        except ImportError:
            _has_constriction = False
            print("[WARN] constriction not installed; skipping rANS baselines (pip install constriction)")

        with open(in_path, 'rb') as rf:
            data = rf.read()

        orig_size = len(data)
        results = {}

        def _bench_codec(name, compress_fn, decompress_fn, out_suffix):
            """Run compression + decompression, measure both times and throughput."""
            try:
                # Compression
                t0 = time.perf_counter()
                compressed = compress_fn(data)
                t_comp = time.perf_counter() - t0
                comp_size = len(compressed)

                # Decompression [Referee Major #6]
                t0 = time.perf_counter()
                decompressed = decompress_fn(compressed)
                t_decomp = time.perf_counter() - t0

                # Verify round-trip
                assert len(decompressed) == orig_size, f"Decompressed size mismatch: {len(decompressed)} vs {orig_size}"

                # Write compressed output
                out_path = out_dir / f"{exp_name}.{out_suffix}"
                with open(out_path, 'wb') as wf:
                    wf.write(compressed)

                results[name] = {
                    'path': str(out_path),
                    'size': comp_size,
                    'time_compress_s': t_comp,
                    'time_decompress_s': t_decomp,
                    'throughput_compress_MBps': (orig_size / 1e6) / max(t_comp, 1e-9),
                    'throughput_decompress_MBps': (orig_size / 1e6) / max(t_decomp, 1e-9),
                }
            except Exception as e:
                results[name] = {'error': str(e)}

        # LZMA (try EXTREME if available, fall back to preset=9)
        _bench_codec(
            'lzma',
            lambda d: lzma.compress(d, preset=9 | getattr(lzma, 'PRESET_EXTREME', 0)),
            lzma.decompress,
            'lzma'
        )

        # ZLIB (max compression level = 9)
        _bench_codec(
            'zlib',
            lambda d: zlib.compress(d, level=9),
            zlib.decompress,
            'zlib'
        )

        # ZSTD at multiple levels [Referee Major #2]
        if _has_zstd:
            for zstd_level in (1, 3, 7, 19):
                _bench_codec(
                    f'zstd_L{zstd_level}',
                    lambda d, lvl=zstd_level: zstd.ZstdCompressor(level=lvl).compress(d),
                    lambda d: zstd.ZstdDecompressor().decompress(d),
                    f'zstd_L{zstd_level}'
                )

        # LZ4 (frame and high-compression) [Referee Major #2]
        if _has_lz4:
            _bench_codec(
                'lz4',
                lambda d: lz4f.compress(d),
                lambda d: lz4f.decompress(d),
                'lz4'
            )
            _bench_codec(
                'lz4_hc',
                lambda d: lz4f.compress(d, compression_level=lz4f.COMPRESSIONLEVEL_MAX),
                lambda d: lz4f.decompress(d),
                'lz4_hc'
            )

        # Brotli baselines
        if _has_brotli:
            _bench_codec(
                'brotli_1',
                lambda d: brotli.compress(d, quality=1),
                lambda d: brotli.decompress(d),
                'brotli_1'
            )
            _bench_codec(
                'brotli_11',
                lambda d: brotli.compress(d, quality=11),
                lambda d: brotli.decompress(d),
                'brotli_11'
            )

        # bzip2 (stdlib)
        _bench_codec(
            'bzip2_9',
            lambda d: bz2.compress(d, compresslevel=9),
            lambda d: bz2.decompress(d),
            'bzip2_9'
        )

        # PPMd (Prediction by Partial Matching)
        if _has_ppmd:
            ppmd_mem = min(256 << 20, max(16 << 20, orig_size * 2))
            _bench_codec(
                'ppmd',
                lambda d: pyppmd.compress(d, max_order=6, mem_size=ppmd_mem, variant='H'),
                lambda d: pyppmd.decompress(d, max_order=6, mem_size=ppmd_mem, variant='H'),
                'ppmd'
            )

        # Order-0 rANS (pure entropy coding baseline)
        if _has_constriction:
            from generate_comparison_table import _rans_o0_compress, _rans_o0_decompress
            _bench_codec(
                'rans_o0',
                _rans_o0_compress,
                _rans_o0_decompress,
                'rans_o0'
            )

        # RNTuple (ROOT) baseline
        if config.get('baseline', {}).get('rntuple', False):
            try:
                rntuple_path = out_dir / f"{exp_name}.root"
                t0 = time.perf_counter()
                file = uproot.recreate(rntuple_path)
                rn_data = np.frombuffer(data_bytes)
                file.mkrntuple("tuple6", {"data": rn_data})
                file.close()
                t_rntuple = time.perf_counter() - t0
                rntuple_size = os.path.getsize(rntuple_path)
                results['rntuple'] = {'path': str(rntuple_path), 'size': rntuple_size,
                                      'time_compress_s': t_rntuple, 'time_decompress_s': float('nan'),
                                      'throughput_compress_MBps': (orig_size / 1e6) / max(t_rntuple, 1e-9),
                                      'throughput_decompress_MBps': float('nan')}
            except Exception as e:
                results['rntuple'] = {'error': str(e)}

        # Print a concise summary [Referee Major #2, #6]
        print("\nBaseline compression results:")
        print(f"  Original size: {orig_size} bytes ({orig_size/1e6:.2f} MB)")
        print(f"  {'Method':<14s}  {'Comp. Size':>12s}  {'Ratio':>7s}  {'Comp. MB/s':>11s}  {'Decomp. MB/s':>13s}  {'Comp. Time':>11s}  {'Decomp. Time':>12s}")
        print(f"  {'-'*90}")
        baseline_order = ['lzma', 'zlib',
                          'zstd_L1', 'zstd_L3', 'zstd_L7', 'zstd_L19',
                          'lz4', 'lz4_hc',
                          'brotli_1', 'brotli_11',
                          'bzip2_9', 'ppmd', 'rans_o0',
                          'rntuple']
        for k in baseline_order:
            r = results.get(k, {})
            if not r:
                continue
            if 'error' in r:
                print(f"  {k.upper():<14s}  ERROR: {r['error']}")
                continue
            size = r['size']
            ratio = orig_size / size if size > 0 else float('inf')
            tc = r.get('time_compress_s', float('nan'))
            td = r.get('time_decompress_s', float('nan'))
            tpc = r.get('throughput_compress_MBps', float('nan'))
            tpd = r.get('throughput_decompress_MBps', float('nan'))
            print(f"  {k.upper():<14s}  {size:>12,d}  {ratio:>7.2f}  {tpc:>11.1f}  {tpd:>13.1f}  {tc:>10.3f}s  {td:>11.3f}s")

        return results

    if args.comparison_baseline_only:
        try:
            os.makedirs(exp_dir, exist_ok=True)
            _run_baseline_comparisons(compress_file_path, exp_dir, name)
            print("\n--comparison-baseline-only complete. Exiting.")
            return
        except Exception as e:
            print(f"[ERROR] Baseline comparison failed: {e}")
            return

    # Online adaptation config
    adapt_config = None
    if args.adapt is not None:
        adapt_config = (args.adapt[0], int(args.adapt[1]))

    boa = BOA(device, str(exp_dir / f"{name}.boa"), model, adapt_config=adapt_config,
             measure_energy=getattr(args, 'measure_energy', False))
    file_format = compress_file_path.suffix.lstrip('.') or 'bin'
    # Compression
    if not args.train_only and not args.decompress_only and not args.evaluate_only:
        print("Starting compression...")
        
        target_compress_path = compress_file_path
        temp_compress_path = None
        
        if vocab_size < 256:
            print(f"Remapping compression input to {vocab_size} vocab size...")
            with open(compress_file_path, 'rb') as f:
                c_data = f.read()
            
            # Check if all bytes in c_data are in vocab
            c_unique = set(c_data)
            if not c_unique.issubset(set(unique_bytes)):
                 print("[ERROR] Compression input contains bytes not seen in training data! Cannot compress.")
                 target_compress_path = None
            else:
                c_arr = np.frombuffer(c_data, dtype=np.uint8)
                c_remapped = lookup[c_arr].tobytes()
                temp_compress_path = exp_dir / f"temp_remapped_{compress_file_path.name}"
                with open(temp_compress_path, 'wb') as f:
                    f.write(c_remapped)
                target_compress_path = temp_compress_path

        if target_compress_path:
            t_start = time.perf_counter()
            # Create BOA that writes into the experiment directory
            boa.compress(
                data_path=str(target_compress_path),
                chunks_count=config.get('compression', {}).get('chunks_count', 1000),
                progress=progress,
            )
            with open(exp_dir / f"{name}.boa", 'rb') as bf:
                boa_size = len(bf.read())
            with open(compress_file_path, 'rb') as rf:
                original_size = len(rf.read())
            compression_ratio = original_size / boa_size if boa_size > 0 else float('inf')

            # Model-inclusive reporting [Referee Major #3]
            model_file = exp_dir / f"{name}_final_model_{precision}.pt"
            model_size_bytes = model_file.stat().st_size if model_file.exists() else 0
            boa_size_with_model = boa_size + model_size_bytes
            ratio_with_model = original_size / boa_size_with_model if boa_size_with_model > 0 else float('inf')
            # Amortisation: how many files of this size needed for model overhead < 1%
            if model_size_bytes > 0:
                amortise_n = int(np.ceil(model_size_bytes / (0.01 * original_size)))
            else:
                amortise_n = 0

            print(f"Compression ratio (excl. model): {compression_ratio:.2f}")
            print(f"Compression ratio (incl. model): {ratio_with_model:.2f}")
            print(f"  Compressed size: {boa_size:,} bytes | Model size: {model_size_bytes:,} bytes")
            print(f"  Combined size:   {boa_size_with_model:,} bytes")
            if amortise_n > 0:
                print(f"  Files needed to amortise model overhead to <1%: {amortise_n}")

            timings['compression'] = time.perf_counter() - t_start
            comp_throughput = (original_size / 1e6) / max(timings['compression'], 1e-9)
            print(f"Compression complete in {timings['compression']:.2f}s ({comp_throughput:.1f} MB/s)")
            
            if temp_compress_path and temp_compress_path.exists():
                temp_compress_path.unlink()

    # Decompression (write decompressed bytes into the experiment directory)
    if not args.train_only and not args.compress_only and not args.evaluate_only:
        print("Starting decompression...")
        t_start = time.perf_counter()
        # BoaFile.decompress() returns the original bytes (which are remapped indices here)
        decompressed_bytes = boa.decompress(progress=progress)
        
        if vocab_size < 256:
             print(f"Remapping decompressed output back to original bytes...")
             # Inverse mapping
             inv_lookup = np.zeros(256, dtype=np.uint8)
             for idx, b in idx_to_byte.items():
                 inv_lookup[idx] = b
             
             d_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8)
             decompressed_bytes = inv_lookup[d_arr].tobytes()

        out_path = exp_dir / f"{name}_decompressed.{file_format}"
        with open(out_path, 'wb') as outf:
            outf.write(decompressed_bytes)
        timings['decompression'] = time.perf_counter() - t_start
        # Decompression throughput [Referee Major #6]
        decomp_throughput = (len(decompressed_bytes) / 1e6) / max(timings['decompression'], 1e-9)
        print(f"Decompression complete in {timings['decompression']:.2f}s ({decomp_throughput:.1f} MB/s)")

        # Optional verification: compare decompressed bytes with original compression input
        # [Referee Minor #6]: Provide bitwise equality statement with checksum
        if verify:
            import hashlib
            # Compare against the bytes we actually compressed (compress_file_path)
            with open(compress_file_path, 'rb') as rf:
                ref_bytes = rf.read()
            same = decompressed_bytes == ref_bytes
            # SHA-256 checksums for paper-grade verification
            ref_hash = hashlib.sha256(ref_bytes).hexdigest()
            dec_hash = hashlib.sha256(decompressed_bytes).hexdigest()
            print(f"\n  === Bitwise Verification ===")
            print(f"  Original SHA-256:     {ref_hash}")
            print(f"  Decompressed SHA-256: {dec_hash}")
            if same:
                print(f"  VERIFY: OK — decompressed output matches input ({len(decompressed_bytes):,} bytes)")
                print(f"  Lossless round-trip: CONFIRMED")
            else:
                # Provide small diagnostic: print sizes and first mismatch position (bounded)
                print("  VERIFY: MISMATCH — decompressed output differs from input")
                if len(decompressed_bytes) != len(ref_bytes):
                    print(f"  Sizes differ: decompressed={len(decompressed_bytes)} vs input={len(ref_bytes)}")
                else:
                    # Find first mismatch up to a cap
                    cap = min(len(decompressed_bytes), 1_000_000)
                    for i in range(cap):
                        if decompressed_bytes[i] != ref_bytes[i]:
                            print(f"  First differing byte at offset {i}: dec={decompressed_bytes[i]} input={ref_bytes[i]}")
                            break

    # Note: configs are stored under experiments/<name>/<name>.yaml when created
    # and can be referenced by experiment name via --config <name>. No copy is necessary.
    if (args.evaluate or args.evaluate_only) and torch.cuda.is_available():
        from evaluator import CompressionEvaluator
        print("Starting evaluation...")
        print("Loading model and data...")

        # Data
        with open(compress_file_path, 'rb') as rf:
            data_bytes = rf.read()
            print(f"Data loaded: {len(data_bytes)/1024/1024:.2f} MB")

        if use_vocab_subset and vocab_size < 256 and lookup is not None:
            print(f"Remapping evaluation data to {vocab_size} vocab size...")
            arr = np.frombuffer(data_bytes, dtype=np.uint8)
            data_bytes = lookup[arr].tobytes()


        # Splits
        n = len(data_bytes)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)
        train_bytes = data_bytes[:train_end]
        val_bytes   = data_bytes[train_end:val_end]
        test_bytes  = data_bytes[val_end:]

        eval_seq_len = 1024
        eval_batch_size = 1
        train_loader = ByteDataloader(train_bytes, seq_len=eval_seq_len, batch_size=eval_batch_size)
        val_loader   = ByteDataloader(val_bytes,   seq_len=eval_seq_len, batch_size=eval_batch_size)
        test_loader  = ByteDataloader(test_bytes,  seq_len=eval_seq_len, batch_size=eval_batch_size)

        # Evaluate & plot all on one figure
        evaluator = CompressionEvaluator(model, device=device)
        os.makedirs(f"experiments/{name}/plots", exist_ok=True)
        curves = evaluator.plot_calibration_curves_multi(
            {"train": train_loader, "val": val_loader, "test": test_loader},
            n_bins=20,
            max_batches=20,            # subset for speed
            savepath=f"experiments/{name}/plots/calibration_all.png",
            quantile_bins=False        # set True for equal-mass bins
        )
        res = evaluator.plot_topk_accuracy(
            test_loader, k_max=20, step=1,
            savepath=f"experiments/{name}/plots/top_k_accuracy.png",
            annotate_ks=(1, 5, 10)
        )
        res = evaluator.plot_confusion_top_bytes(test_loader, top_n=20, normalize="true",
                                savepath=f"experiments/{name}/plots/byte_confusion_matrix.png")
        # Also plot original vs decompressed comparison for first few columns to show bit-exactness
        try:
            decompressed_path = exp_dir / f"{name}_decompressed.{file_format}"
            if decompressed_path.exists():
                evaluator.plot_bit_exact_columns(
                    original_file=str(compress_file_path),
                    decompressed_file=str(decompressed_path),
                    num_cols=4,
                    dtype='float32',
                    max_rows=2000,
                    savepath=f"experiments/{name}/plots/bit_exact_columns.png",
                )
                # Also generate formal bitwise verification report [Referee Minor #6]
                CompressionEvaluator.verify_bitwise(
                    original_file=str(compress_file_path),
                    decompressed_file=str(decompressed_path),
                    report_path=f"experiments/{name}/plots/bitwise_verification.json",
                )
            else:
                print(f"[INFO] Decompressed file not found at {decompressed_path}; skipping bit-exact columns plot.")
        except Exception as e:
            print(f"[WARN] Failed to generate bit-exact columns plot: {e}")
        print("Evaluation complete.")
    elif not torch.cuda.is_available() and (args.evaluate or args.evaluate_only):
        print("[WARN] Evaluation requires CUDA; skipping evaluation as no CUDA device is available.")
        
    if args.show_timings:
        print('\nTimings:')
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")
        # Clarification per Referee Minor #4:
        print("  Note: Compression/decompression times include model inference,")
        print("  entropy coding, and in-memory I/O. File read/write times are")
        print("  reported separately as 'read_bytes'. CPU-GPU transfer overhead")
        print("  is included in the respective stage timings.")


if __name__ == '__main__':
    main()


