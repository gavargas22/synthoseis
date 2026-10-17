# Physical rock physics (default) and `--legacy-rock-physics`

Pointwise chain per layer (`synthoseis-rpm/src/kernels/physical.rs`):
Athy porosity → hydrostatic effective pressure → soft-sand dry frame
(Hertz-Mindlin + modified lower Hashin-Shtrikman) → Batzle-Wang (1992)
brine / gas / live oil → Wood mix (S_hc 0.8) → Gassmann. Sand and shale are
mixed by net-to-gross as before. `--legacy-rock-physics` restores the
77131867 depth-trend polynomials byte for byte.

| Parameter | Default | Documented range | Source |
|---|---|---|---|
| Sand Athy φ0 / c | 0.49 / 0.27 km⁻¹ | flags `--sand-phi0`, `--sand-athy-c` | Sclater & Christie 1980 |
| Shale Athy φ0 / c | 0.63 / 0.51 km⁻¹ | flags `--shale-phi0`, `--shale-athy-c` | Sclater & Christie 1980 |
| Sand φ_c / coordination n | 0.40 / 9 | n 5–14 | Mavko et al. 2009 |
| Shale clay fraction | 0.30 (**fitted**, D1) | 0.2–0.9 | calibration |
| Shale coordination n | 12 (**fitted**, D1) | 5–14 | calibration; Mavko et al. 2009, doi:10.1017/CBO9780511626753 |
| Shale φ_c | 0.65 (fitted, D1) | – | calibration |
| Temperature | 4 °C + 30 °C/km | flags | – |
| Brine salinity | 0.035 | flag | – |
| Gas gravity / oil API / GOR | 0.6 / 32 / 100 L/L | flags | – |

The shale clay fraction and coordination number are fit parameters chosen
so that shale Vp ≈ 3000 m/s at 2 km and the sand/shale impedance contrast
stays near the legacy model. They are not measured contact counts.

Live-oil density is B&W eq. 24, ρ_G = (ρ0 + 0.0012 G R_G)/B0, with the
reservoir taken at saturation. The eq. 18–19 pressure and temperature
adjustment is not applied on top, because B0 already includes the
temperature.

Known effects: shallow gas sand reaches ~650 m/s near the seabed. The
depth-staircase warning uses the model's own minimum velocity. Slower
sediment moves more of the deep section past the fixed-length time axis
(`default_twt_samples`).
