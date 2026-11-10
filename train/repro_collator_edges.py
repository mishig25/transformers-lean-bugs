"""
Low-priority collator edge cases found by the Lean model (Collators.lean).

(1) DataCollatorForLanguageModeling(mask_replace_prob=0.8, random_replace_prob=0.2) passes validation
    (0.8 + 0.2 <= 1) but random_replace_prob / (1 - mask_replace_prob) = 0.2 / 0.19999999999999996
    = 1.0000000000000002 in float64.  torch path: torch.full(..) is float32 -> rounds to 1.0, fine.
    numpy path (return_tensors="np"): binomial(1, p>1) raises.
(2) DataCollatorWithFlattening with an empty example: labels get one more element than input_ids.
"""

import numpy as np
import torch

from transformers import DataCollatorForLanguageModeling, DataCollatorWithFlattening


class Tok:
    mask_token = "[MASK]"
    mask_token_id = 3
    pad_token = "[PAD]"
    pad_token_id = 0
    padding_side = "right"
    is_fast = True

    def __len__(self):
        return 32

    def get_special_tokens_mask(self, val, already_has_special_tokens=True):
        return [0] * len(val)

    def convert_tokens_to_ids(self, t):
        return 3


for p_m, p_r in [(0.8, 0.2), (0.9, 0.1), (0.8, 0.1)]:
    for rt in ["pt", "np"]:
        c = DataCollatorForLanguageModeling(Tok(), mask_replace_prob=p_m, random_replace_prob=p_r, return_tensors=rt)
        try:
            c([[5, 6, 7, 8, 9, 10, 11, 12]] * 4)
            r = "ok"
        except Exception as e:
            r = f"{type(e).__name__}: {e}"
        print(f"mask_replace_prob={p_m} random_replace_prob={p_r} return_tensors={rt}: {r}")

b = DataCollatorWithFlattening()([{"input_ids": [5, 6, 7]}, {"input_ids": []}, {"input_ids": [8, 9]}])
print("flattening with empty example: input_ids", b["input_ids"].tolist(), "labels", b["labels"].tolist())
