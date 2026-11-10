"""Video: processor(text, videos) placeholder count vs _get_num_multimodal_tokens(video_sizes)."""

import sys

import torch

from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata


CASES = [(F, h, w) for F in (2, 3, 5, 7, 8, 16) for (h, w) in ((360, 640), (480, 854), (720, 1280), (240, 320))]

for repo in sys.argv[1:]:
    proc = AutoProcessor.from_pretrained(repo)
    vt = proc.video_token
    vid = proc.video_token_id
    bad = []
    for F, h, w in CASES:
        video = torch.randint(0, 256, (F, 3, h, w), dtype=torch.uint8)
        meta = VideoMetadata(total_num_frames=F, fps=2.0, frames_indices=list(range(F)), duration=F / 2.0)
        try:
            out = proc(text=[f"Describe {vt}"], videos=[video], video_metadata=[meta], do_sample_frames=False,
                       return_tensors="pt")
        except Exception as e:  # noqa: BLE001
            bad.append((F, h, w, f"processor error {type(e).__name__}: {str(e)[:100]}"))
            continue
        actual = int((out["input_ids"] == vid).sum()) or int((out["input_ids"] == proc.image_token_id).sum())
        try:
            pred = proc._get_num_multimodal_tokens(video_sizes=[[F, h, w]])["num_video_tokens"][0]
        except Exception as e:  # noqa: BLE001
            bad.append((F, h, w, f"mm error {type(e).__name__}: {str(e)[:100]}"))
            continue
        if actual != pred:
            bad.append((F, h, w, f"actual={actual} predicted={pred}"))
    print(f"== {repo}: {len(CASES) - len(bad)}/{len(CASES)} agree")
    for b in bad[:12]:
        print("   ", b)
