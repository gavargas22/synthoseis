"""Plot the raw volumes written by `cargo run --example faults_demo`.

    python plot_faults_demo.py DEMO_DIR OUT_PREFIX

Writes OUT_PREFIX_slices.png (inline/crossline age + labels with fault-mask
overlay) and OUT_PREFIX_stack.png (angle-stack slices, unfaulted vs faulted
labels, fault mask overlay).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(d, name, dtype, shape):
    return np.fromfile(d / name, dtype=dtype).reshape(shape)


def overlay(ax, mask2d):
    m = np.ma.masked_where(mask2d.T == 0, mask2d.T)
    ax.imshow(m, cmap="autumn", alpha=0.85, aspect="auto", interpolation="nearest")


def main():
    d = Path(sys.argv[1])
    prefix = sys.argv[2]
    meta = json.loads((d / "meta.json").read_text())
    shape = tuple(meta["shape"])
    age = load(d, "age.f32", "<f4", shape)
    labels = load(d, "labels.u8", "u1", shape)
    unf = load(d, "labels_unfaulted.u8", "u1", shape)
    mask = load(d, "fault_mask.u8", "u1", shape)
    stack = load(d, "angle_stack.f32", "<f4", shape)

    counts_i = mask.sum(axis=(1, 2))
    counts_j = mask.sum(axis=(0, 2))
    il = int(np.argmax(counts_i))
    xl = int(np.argmax(counts_j))
    title = (f"seed={meta['seed']} shape={shape} faults inserted="
             f"{meta['inserted']}/{meta['requested']} fault voxels={meta['fault_voxels']}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (name, sl) in enumerate([(f"inline {il}", np.s_[il, :, :]),
                                      (f"crossline {xl}", np.s_[:, xl, :])]):
        ax = axes[row, 0]
        ax.imshow(np.mod(age[sl], 2.0).T, cmap="viridis", aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"faulted layer-cake age mod 2 — {name}")
        ax = axes[row, 1]
        ax.imshow(np.mod(age[sl], 2.0).T, cmap="gray", aspect="auto",
                  interpolation="nearest")
        overlay(ax, mask[sl])
        ax.set_title(f"age + fault mask — {name}")
        ax = axes[row, 2]
        ax.imshow(labels[sl].T, cmap="tab10", vmin=0, vmax=9, aspect="auto",
                  interpolation="nearest")
        overlay(ax, mask[sl])
        ax.set_title(f"pipeline labels (faulted) + mask — {name}")
    for ax in axes.flat:
        ax.set_xlabel("trace")
        ax.set_ylabel("sample (k)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{prefix}_slices.png", dpi=110)

    k_mid = int(np.argmax(mask.sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    v = np.percentile(np.abs(stack), 99) or 1.0
    axes[0].imshow(unf[il].T, cmap="tab10", vmin=0, vmax=9, aspect="auto",
                   interpolation="nearest")
    axes[0].set_title(f"labels WITHOUT faults — inline {il}")
    axes[1].imshow(stack[il].T, cmap="seismic", vmin=-v, vmax=v, aspect="auto")
    axes[1].set_title(f"angle stack (faulted) — inline {il}")
    axes[2].imshow(stack[il].T, cmap="gray", vmin=-v, vmax=v, aspect="auto")
    overlay(axes[2], mask[il])
    axes[2].set_title("angle stack + fault mask")
    axes[3].imshow(np.mod(age[:, :, k_mid], 2.0).T, cmap="gray", aspect="auto",
                   interpolation="nearest")
    overlay(axes[3], mask[:, :, k_mid])
    axes[3].set_title(f"depth slice k={k_mid}: age + fault mask")
    axes[3].set_xlabel("inline")
    axes[3].set_ylabel("crossline")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{prefix}_stack.png", dpi=110)
    print(f"wrote {prefix}_slices.png {prefix}_stack.png (il={il}, xl={xl}, k={k_mid})")


if __name__ == "__main__":
    main()
