#!/usr/bin/env python3
"""Export the Voxtral TTS 8-bit pack (K-90): LLM and flow-matching transformer 8-bit, group 64, affine; codec bf16.

Recipe: start from mlx-community/Voxtral-4B-TTS-2603-mlx-bf16 at a pinned revision and quantize exactly the matrices
that mlx-community/Voxtral-4B-TTS-2603-mlx-6bit quantizes (the keys with `.scales` in that pack), so the layer selection
is the one of the 4-bit and 6-bit packs; every other tensor is copied unchanged. The config gets `quantization` and
`quantization_config` = {group_size 64, bits 8, mode affine}. Not a dependency of the code, the tests or the build
(ASK-12): run it once in a throwaway environment with `mlx` installed.

Usage: python3 -I export_tts_8bit.py <bf16 pack dir> <6-bit pack dir> <output dir>
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

SOURCE_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"
SOURCE_REVISION = "dd85c02adbae551f5bb29ded35ee60ccdfb90927"
LAYOUT_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit"
LAYOUT_REVISION = "11682c79d1bdda7a500fa146276459e3c6e253ad"
GROUP_SIZE, BITS, MODE = 64, 8, "affine"
SHARD_BYTES = 5_000_000_000


def load_dir(path: Path) -> dict:
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    return weights


def main() -> None:
    source, layout, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=False)
    weights = load_dir(source)
    layout_keys = set(mx.load(str(layout / "model.safetensors")).keys()) if (layout / "model.safetensors").exists() \
        else set(load_dir(layout).keys())
    quantized_modules = sorted(k[: -len(".scales")] for k in layout_keys if k.endswith(".scales"))

    result = dict(weights)
    for module in quantized_modules:
        w = result.pop(f"{module}.weight")
        q, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, mode=MODE)
        result[f"{module}.weight"], result[f"{module}.scales"], result[f"{module}.biases"] = q, scales, biases

    expected = {k for k in layout_keys}
    if set(result) != expected:
        raise SystemExit(f"key set differs from the 6-bit layout: missing {sorted(expected - set(result))[:5]}, "
                         f"extra {sorted(set(result) - expected)[:5]}")

    # Shards in key order, index like mlx-community's
    shards, current, size = [], {}, 0
    for key in sorted(result):
        nbytes = result[key].nbytes
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = {}, 0
        current[key] = result[key]
        size += nbytes
    shards.append(current)
    weight_map, total = {}, 0
    for i, shard in enumerate(shards):
        name = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), shard, metadata={"format": "mlx"})
        for key in shard:
            weight_map[key] = name
        total += sum(v.nbytes for v in shard.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=4))

    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS, "mode": MODE}
    config["quantization_config"] = dict(config["quantization"])
    (out / "config.json").write_text(json.dumps(config, indent=4))
    for name in ["params.json", "tekken.json"]:
        shutil.copy2(source / name, out / name)
    shutil.copytree(source / "voice_embedding", out / "voice_embedding")

    print(f"source {SOURCE_REPO}@{SOURCE_REVISION}; layout {LAYOUT_REPO}@{LAYOUT_REVISION}")
    print(f"quantized modules {len(quantized_modules)} ({BITS}-bit g{GROUP_SIZE} {MODE}); tensors {len(result)}")
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.suffix in {".safetensors", ".json"}:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            print(f"{f.relative_to(out)} {f.stat().st_size} {digest}")


if __name__ == "__main__":
    main()
