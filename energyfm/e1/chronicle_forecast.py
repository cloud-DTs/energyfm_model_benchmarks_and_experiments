"""Run two zero-shot Chronicle variants against the retained E1 Chronos evidence."""

import json
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_model

from energyfm.assets import require_model

from .chronicle_model import Chronicle, ChronicleConfig
from .chronicle_tokenizer import ChronicleTokenizer


def context_text(case):
    """Represent the E1 hourly weather values in compact, chronological tables.

    Inputs: A forecast case provides ordered past and future weather channels from which
        to build the prompt.
    Outputs: A text table with named columns and hourly rows, ready for the checkpoint
        tokenizer.
    Purpose: Add weather context in the form accepted by Chronicle while preserving the
        same underlying weather values.
    """
    names = list(case.past)
    sections = [
        "Archived weather forecasts. One row per hour. "
        "Past rows align to the measured history. Future rows start with forecast lead 1.",
        ",".join(names),
    ]
    for label, fields in (("Past", case.past), ("Future", case.future)):
        sections.append(label)
        rows = zip(*(fields[name] for name in names), strict=True)
        sections.extend(
            ",".join("0" if value == 0 else format(value, ".4g") for value in row)
            for row in rows
        )
    return "\n".join(sections)


class ChronicleForecaster:
    def __init__(self, device):
        """Load the pinned local Stage 2 weights and tokenizer onto the selected device.

        Inputs: The selected device determines placement. The local checkpoint directory
            supplies the config, trained weights and tokenizer.
        Outputs: None. Initializes the forecaster so later encoding and prediction use
            the same checkpoint.
        Purpose: Load the model and tokenizer once per command so both variants share
            them.
        """
        snapshot = require_model("chronicle")
        checkpoint = snapshot / "stage-2"
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        self.config = ChronicleConfig(**config)
        self.model = Chronicle(self.config)
        load_model(self.model, str(checkpoint / "model.safetensors"), strict=True)
        self.model.to(device).eval()
        self.tokenizer = ChronicleTokenizer.from_directory(snapshot / "tokenizer")
        self.device = device

    def encode(self, text):
        """Return the required BOS token followed by the encoded weather prompt.

        Inputs: The prompt text supplies optional weather context. An empty string
            represents the history-only variant.
        Outputs: A list of token IDs beginning with BOS, ready to pass beside the
            numeric history.
        Purpose: Give both variants the same required initial token and the checkpoint's
            expected tokenization.
        """
        return [self.tokenizer.get_bos_token_id(), *self.tokenizer.encode(text)]

    @torch.inference_mode()
    def predict_batch(self, histories, horizon, tokens):
        """Return one median forecast per history, in the original batch order.

        Inputs: Same-length numeric histories and prompt token lists form a batch. The
            horizon determines how many next-patch predictions are needed.
        Outputs: A NumPy array of finite medians shaped [case, horizon] in input order
            for later slicing and scoring.
        Purpose: Use the same padding, mask, quantile choice and autoregressive rollout
            for both variants.
        """
        patch = self.config.patch_len
        recorded = torch.as_tensor(histories, dtype=torch.float32, device=self.device)
        batch_size, history_length = recorded.shape
        # Left-pad to a whole patch so its last observation remains the forecast origin.
        padding = (-history_length) % patch
        ids = torch.tensor(tokens, device=self.device)
        remaining = horizon
        blocks = []
        while remaining:
            padded = torch.nn.functional.pad(recorded, (padding, 0))
            valid = torch.arange(padded.shape[1], device=self.device) >= padding
            values = padded.reshape(batch_size, -1, patch)
            mask = valid.reshape(1, -1, patch).expand_as(values)
            if ids.shape[1] + values.shape[1] > self.config.sequence_len:
                raise ValueError(
                    "Weather text and energy history exceed the Chronicle context limit."
                )
            _, quantiles = self.model(
                ids, ts_patches=values, ts_mask_in=mask, return_text_logits=False
            )
            count = min(remaining, patch)
            median = quantiles[:, -1, :count, self.config.num_quantiles // 2]
            blocks.append(median)
            remaining -= count
            if remaining:
                # Advance by a full predicted patch. Later steps can use these
                # predictions as history without consuming observed future targets.
                recorded = torch.cat((recorded, median), dim=1)
        forecasts = torch.cat(blocks, dim=1).cpu().numpy()
        if not np.isfinite(forecasts).all():
            raise ValueError("Chronicle returned a non-finite median forecast.")
        return forecasts


def forecast_cases(model, cases, tokens, batch_size, progress_label=None):
    """Batch shape-compatible cases and return the complete forecast horizon per case.

    Inputs: The forecaster, ordered cases and matching token lists define the requests.
        The batch limit bounds memory use and the optional progress label names status
        messages.
    Outputs: The complete forecast per case in original order, plus batch and
        retry statistics for provenance.
    Purpose: Batch shape-compatible cases efficiently and recover from GPU memory
        pressure while preserving case pairing.
    """
    groups = defaultdict(list)
    # Equal shapes let us batch without padding prompts. Keep case indices to
    # restore the original order when forecasts are scattered back for scoring.
    for index, (case, ids) in enumerate(zip(cases, tokens, strict=True)):
        groups[len(case.history), case.horizon, len(ids)].append(index)

    forecasts = np.empty((len(cases), cases[0].horizon), dtype=np.float32)
    completed = 0
    batch_limit = batch_size
    max_batch_used = 0
    oom_retries = 0
    for (_, horizon, _), indices in groups.items():
        offset = 0
        while offset < len(indices):
            batch = indices[offset : offset + batch_limit]
            histories = np.stack([cases[index].history for index in batch])
            try:
                values = model.predict_batch(
                    histories, horizon, [tokens[index] for index in batch]
                )
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise
            else:
                forecasts[batch] = values
                offset += len(batch)
                max_batch_used = max(max_batch_used, len(batch))
                previous = completed
                completed += len(batch)
                if progress_label and (
                    previous // 100 != completed // 100 or completed == len(cases)
                ):
                    print(
                        f"{progress_label}: {completed}/{len(cases)} cases", flush=True
                    )
                continue
            # The failed call has unwound, so its temporary tensors can be released.
            torch.cuda.empty_cache()
            batch_limit = max(1, len(batch) // 2)
            oom_retries += 1
            if progress_label:
                print(
                    f"{progress_label}: reducing batch size to {batch_limit} after CUDA OOM.",
                    flush=True,
                )
    return forecasts, {
        "batch_size_requested": batch_size,
        "max_batch_size_used": max_batch_used,
        "oom_retries": oom_retries,
    }
