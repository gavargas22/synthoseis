# Synthoseis Rust workspace

Skeleton for the Rust rewrite of [synthoseis](https://github.com/gavargas22/synthoseis).
The Python tree at the repository root stays intact; this workspace lives under `rust/`.

## Crates

| Crate | Role |
|-------|------|
| `synthoseis` | CLI binary (`synthoseis --help`, `synthoseis run`) |
| `synthoseis-core` | Config, RNG, job partition, single-worker path stubs |
| `synthoseis-io` | **MDIO-only** working store + deliverable API stubs |
| `synthoseis-geo` | Future geology / horizons port |
| `synthoseis-closures` | Future closure / trap geometry port |
| `synthoseis-seismic` | Future seismic modeling port |
| `synthoseis-rpm` | Future rock-physics model port |
| `synthoseis-gpu` | Future GPU acceleration port |

## I/O lock: MDIO only

Working volumes and deliverables use an **MDIO-intent** on-disk store (chunked array + JSON attrs).
There is no seismic MDIO crate on crates.io (`mdio` is Ethernet PHY — do not use it).
MDIO ([mdio.dev](https://mdio.dev)) is Zarr-based.

This skeleton writes a **minimal Zarr-v2-like directory layout** (`.zgroup`, `.zarray`, `.zattrs`, chunk files)
with dimensions `inline` / `crossline` / `time` and attrs `digi`, `seed`, `units`.
Full MDIO / Python interop is a follow-up; prefer a maintained Zarr crate (`zarrs` or similar) when wiring production I/O.

## Parity harness

`synthoseis-core` includes a fixed-seed golden comparison **placeholder**.
Python baseline vs Rust MAE/IoU metrics come later — do not treat the stub as numeric parity.

## Single-worker path

CLI/`synthoseis-core` stubs a **one local worker** run path. No cloud orchestration yet.

## Develop

```bash
cd rust
cargo check
cargo test
cargo run -p synthoseis -- --help
cargo run -p synthoseis -- run
```

CI: `.github/workflows/rust-ci.yml` runs `cargo check` + `cargo test` in `rust/` on push/PR.

## Next ports (out of scope here)

- Geology / closures / seismic / RPM algorithms
- Real MDIO Python round-trip
- Multi-worker / cloud job partition
