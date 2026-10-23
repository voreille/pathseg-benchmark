"""Print the val mIoU curve of a local wandb run, without the wandb server.

wandb >= 0.2x no longer ships a Python reader for ``run-*.wandb`` files. They are
protobuf ``Record``s in leveldb-log framing (7-byte file header, 32 KiB blocks,
7-byte record headers: crc32, length, type FULL/FIRST/MIDDLE/LAST).

    uv run python experiments/read_wandb_history.py runs/wandb/run-*-<id>/run-<id>.wandb
"""
import json
import struct
import sys

from wandb.proto import wandb_internal_pb2 as pb

BLOCK, HDR = 32768, 7


def records(path):
    data = open(path, "rb").read()
    pos, buf = 7, b""  # 7-byte file header ":W&B" + magic + version
    while pos + HDR <= len(data):
        left = BLOCK - (pos % BLOCK)
        if left < HDR:
            pos += left
            continue
        _crc, length, typ = struct.unpack("<IHB", data[pos : pos + HDR])
        pos += HDR
        chunk = data[pos : pos + length]
        pos += length
        if typ == 0 and length == 0:
            continue
        if typ == 1:
            yield chunk
        elif typ == 2:
            buf = chunk
        elif typ == 3:
            buf += chunk
        elif typ == 4:
            yield buf + chunk


rows = []
for raw in records(sys.argv[1]):
    r = pb.Record()
    try:
        r.ParseFromString(raw)
    except Exception:
        continue
    if r.WhichOneof("record_type") == "history":
        rows.append({i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item})

val = [r for r in rows if any(k.startswith("val") and "miou" in k for k in r)]
print("history rows", len(rows), "val rows", len(val))
for r in val:
    print(r.get("trainer/global_step"), {k: round(v, 4) for k, v in r.items() if "miou" in k or "mf1" in k})
if val:
    last = val[-1]
    print("per-class IoU at last val:", {k.split("_")[-1]: round(v, 3) for k, v in last.items() if "_iou_" in k})
