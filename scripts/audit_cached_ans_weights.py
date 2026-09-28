#!/usr/bin/env python3
"""Read cached ANS weights on CPU; synthetic inputs establish compatibility only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main(inventory, output):
    import torch
    import torchvision
    from model import Local_IL_Policy, Neural_SLAM_Module, RL_Policy

    torch.set_num_threads(1)
    torch.manual_seed(2026)
    entries = json.loads(inventory.read_text())["checkpoints"]
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", input_inventory_sha256=sha(inventory),
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                   (Path(__file__), ROOT / "model.py", ROOT / "utils/model.py",
                                    ROOT / "utils/distributions.py")},
                    torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                    synthetic_inputs=True, training_performed=False,
                    navigation_efficacy_proven=False,
                    constructor_note="ResNet download disabled during construction; strict state_dict load replaces every parameter and buffer.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    original_resnet = torchvision.models.resnet18
    for entry in entries:
        row = dict(path=entry["path"], oid=entry["oid"])
        try:
            oid = entry["oid"]
            cached = ROOT / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
            if sha(cached) != oid:
                raise ValueError("Cached payload hash mismatch")
            state = torch.load(cached, map_location="cpu", weights_only=True)
            kind = Path(entry["path"]).suffix
            with patch("torchvision.models.resnet18", side_effect=lambda *a, **kw: original_resnet(weights=None)):
                if kind == ".local":
                    model = Local_IL_Policy((3, 128, 128), 3, recurrent=True, hidden_size=512)
                elif kind == ".slam":
                    args = SimpleNamespace(device=torch.device("cpu"), frame_height=128, frame_width=128,
                        map_resolution=5, map_size_cm=2400, global_downscaling=2, vision_range=64,
                        use_pose_estimation=2, pretrained_resnet=False, num_processes=1, slam_batch_size=1)
                    model = Neural_SLAM_Module(args)
                elif kind == ".global":
                    # The original constructor dispatches by the space class name and shape only.
                    space = type("Box", (), {"shape": (2,)})()
                    model = RL_Policy((8, 240, 240), space,
                        base_kwargs=dict(recurrent=False, hidden_size=256, downscaling=2))
                else:
                    raise ValueError("Unexpected checkpoint kind")
            model.load_state_dict(state, strict=True)
            row["strict_state_load_passed"] = True
            model.eval()
            with torch.inference_mode():
                if kind == ".local":
                    values = model(torch.zeros(1, 3, 128, 128), torch.zeros(1, 512),
                                   torch.ones(1), torch.zeros(1, 2, dtype=torch.long))
                elif kind == ".slam":
                    rgb = torch.zeros(1, 3, 128, 128)
                    values = model(rgb, rgb, torch.zeros(1, 3), torch.zeros(1, 240, 240),
                                   torch.zeros(1, 240, 240), torch.tensor([[6., 6., 0.]]), build_maps=True)
                else:
                    values = model.act(torch.zeros(1, 8, 240, 240), torch.zeros(1, 1),
                                       torch.ones(1, 1), extras=torch.zeros(1, 1, dtype=torch.long),
                                       deterministic=True)
            row["output_shapes"] = [list(v.shape) if isinstance(v, torch.Tensor) else None for v in values]
            row["all_outputs_finite"] = all(bool(torch.isfinite(v).all()) for v in values if isinstance(v, torch.Tensor))
            if not row["all_outputs_finite"]:
                raise ValueError("Nonfinite network output")
            row["status"] = "passed"
            del model, state, values
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
        (output / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(row["path"], row["status"], row.get("error", ""), flush=True)
    manifest.update(status="complete", checkpoints=len(rows), passed=sum(r["status"] == "passed" for r in rows))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "verification.json").write_text(json.dumps(dict(**manifest, checks=rows), indent=2) + "\n")
    if manifest["passed"] != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.inventory, args.output)
