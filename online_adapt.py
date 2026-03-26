"""
Online head adaptation for BOA compression.

During compression/decompression, the prediction head is continuously
fine-tuned on the bytes being processed.  Both encoder and decoder
see the same bytes and perform identical gradient updates, keeping
their models perfectly synchronized.

This is the neural equivalent of cmix's online model adaptation —
the single most impactful technique in compression.

Design:
  - Backbone (Mamba/LSTM/etc.) stays FROZEN
  - Only the head (Linear → SiLU/ReLU → Linear, ~50K params for d=128)
    is adapted
  - Adaptation happens BETWEEN batches (not inside the streaming loop)
    so we don't fight @torch.inference_mode or Mamba's InferenceParams
  - After each batch of N streams finishes, we take the raw bytes that
    were just compressed/decompressed, run ONE forward pass through the
    full model, and do K gradient steps on the head only
  - Both encoder and decoder process the same bytes → identical updates

Usage in boa.py (between batch loops):
    adapter = HeadAdapter(model, lr=2e-4)
    for batch_start in range(0, n_chunks, gpu_streams):
        # ... compress batch ...
        adapter.adapt_on_batch(batch_bytes, device=device)
"""

import torch
import torch.nn.functional as F
import os

# Environment variable to enable/disable online adaptation
_ENABLE_ADAPT = os.environ.get("BOA_ADAPT", "1") != "0"


class HeadAdapter:
    """Manages online adaptation of the prediction head during compression.

    Between batches, takes the compressed/decompressed bytes and runs
    K gradient steps on the head using the full model forward pass.
    The backbone stays frozen — only head parameters receive gradients.
    """

    def __init__(self, model, lr: float = 2e-4, adapt_steps: int = 1,
                 max_grad_norm: float = 1.0, max_seq_len: int = 4096):
        self.model = model
        self.lr = lr
        self.adapt_steps = adapt_steps
        self.max_grad_norm = max_grad_norm
        self.max_seq_len = max_seq_len  # cap sequence length for adaptation to save memory

        # Identify head parameters
        self.head = model.head
        self.head_params = list(self.head.parameters())

        # Save original head state so we can restore after compress/decompress
        # (ensures encoder and decoder both start from the same checkpoint)
        self._original_state = {k: v.clone() for k, v in self.head.state_dict().items()}

        # Freeze everything except head
        for param in model.parameters():
            param.requires_grad_(False)
        for param in self.head_params:
            param.requires_grad_(True)

        # Deterministic SGD — no momentum so encoder/decoder produce
        # bitwise identical updates given the same data
        self.optimizer = torch.optim.SGD(self.head_params, lr=lr)

    def restore_head(self):
        """Restore head to original (pre-adaptation) state.
        Call after compress() finishes so that decompress() starts identically.
        """
        self.head.load_state_dict(self._original_state)
        self._de_inference_head()
        # Reset optimizer state
        self.optimizer = torch.optim.SGD(self.head_params, self.lr)

    def _de_inference_head(self):
        """Replace any inference-mode-tainted parameters with fresh copies.

        compress_GPU / decompress_GPU use @torch.inference_mode(), which
        permanently marks parameters used in that context as inference
        tensors.  Such tensors cannot participate in autograd.
        We fix this by replacing them with non-inference clones.
        """
        for name, module in self.head.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.is_inference():
                    module.weight = torch.nn.Parameter(
                        module.weight.data.clone(), requires_grad=True
                    )
                if module.bias is not None and module.bias.is_inference():
                    module.bias = torch.nn.Parameter(
                        module.bias.data.clone(), requires_grad=True
                    )
        # Refresh param list and optimizer after parameter replacement
        self.head_params = list(self.head.parameters())
        for p in self.head_params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.SGD(self.head_params, self.lr)

    def _get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone only (frozen, no grad) to get hidden states.

        x: [B, L] long tensor of byte sequences
        Returns: [B, L, D] plain tensor (no grad, no inference mode)
        """
        # Use no_grad + explicit inference_mode(False) to ensure we get
        # plain tensors even if Mamba blocks have cached inference tensors
        with torch.no_grad(), torch.inference_mode(False):
            h = self.model.embedding(x)  # [B, L, D]

            # Check if model has inference_params-style forward
            if hasattr(self.model, '_backbone_name') and \
               self.model._backbone_name in ('mamba', 'mambav1', 'mamba2'):
                # For Mamba models, use the full forward pass through blocks
                # (NOT streaming mode — full sequence parallel)
                for blk in self.model.blocks:
                    h = blk(h)  # No inference_params → runs in training/parallel mode
            else:
                for blk in self.model.blocks:
                    h = blk(h)

            # Apply final_norm if present
            if hasattr(self.model, 'final_norm'):
                h = self.model.final_norm(h)

        return h.detach().clone()  # clone ensures plain tensor, no inference mode residue

    def adapt_on_batch(self, byte_sequences: list, device: str = "cuda"):
        """Adapt the head on a batch of byte sequences.

        byte_sequences: list of 1D byte tensors or numpy arrays (the raw
                        bytes that were just compressed/decompressed).
        device: torch device string.

        Must be called OUTSIDE @torch.inference_mode() — we explicitly
        disable inference mode and fix tainted parameters here.
        """
        if not byte_sequences or self.adapt_steps <= 0:
            return

        # Fix parameters tainted by @torch.inference_mode() in codec
        self._de_inference_head()

        # Explicitly exit inference mode for the adaptation step
        with torch.inference_mode(False):
            self._adapt_on_batch_impl(byte_sequences, device)

    def _adapt_on_batch_impl(self, byte_sequences: list, device: str):
        """Inner implementation of adapt_on_batch (runs outside inference mode)."""
        # Convert to tensor and truncate to max_seq_len for memory efficiency
        def _seq_len(s):
            if isinstance(s, torch.Tensor):
                return s.numel()
            return len(s) if hasattr(s, '__len__') else int(s.shape[-1])
        max_len = min(max(_seq_len(s) for s in byte_sequences), self.max_seq_len)
        batch_size = len(byte_sequences)

        # Build padded tensor [B, L]
        x = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(byte_sequences):
            if isinstance(seq, torch.Tensor):
                s = seq.to(device).flatten()[:max_len]
            elif isinstance(seq, (bytes, bytearray)):
                s = torch.tensor(list(seq[:max_len]), dtype=torch.long, device=device)
            else:
                import numpy as np
                s = torch.from_numpy(np.frombuffer(seq, dtype=np.uint8)[:max_len].copy()).to(device)
            x[i, :len(s)] = s.to(torch.long)

        # Input: x[:, :-1], Target: x[:, 1:]
        x_in = x[:, :-1]
        y = x[:, 1:]

        if x_in.shape[1] < 1:
            return

        # Get backbone features (frozen, no grad)
        h = self._get_backbone_features(x_in)  # [B, L-1, D]

        # Adaptation steps: gradient only flows through the head
        self.head.train()
        for _ in range(self.adapt_steps):
            logits = self.head(h)  # [B, L-1, V]
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.head_params, self.max_grad_norm)
            self.optimizer.step()

        # Back to eval for inference
        self.head.eval()

    def adapt_on_raw_bytes(self, raw_bytes: bytes, chunk_len: int,
                           device: str = "cuda", max_chunks: int = 64):
        """Convenience: adapt on raw bytes split into chunks.

        Takes a contiguous byte buffer and splits it into chunks for adaptation.
        Useful when called from boa.py.
        """
        import numpy as np
        total = len(raw_bytes)
        n_chunks = min((total + chunk_len - 1) // chunk_len, max_chunks)

        seqs = []
        for i in range(n_chunks):
            s = i * chunk_len
            e = min(s + chunk_len, total)
            seqs.append(torch.tensor(list(raw_bytes[s:e]), dtype=torch.long))

        self.adapt_on_batch(seqs, device=device)

