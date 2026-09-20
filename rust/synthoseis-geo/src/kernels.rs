/// Least-squares plane fit: `Z = aX + bY + c`.
///
/// Port of `Horizons.fit_plane_lsq` (`np.linalg.lstsq` on design matrix `[X Y 1]`).
pub fn fit_plane_lsq(xyz: &[[f64; 3]]) -> [f64; 3] {
    assert!(
        xyz.len() >= 3,
        "fit_plane_lsq needs at least 3 points, got {}",
        xyz.len()
    );
    // Normal equations for min ||G θ - z|| with G = [x y 1], θ = [a b c].
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atz = [0.0_f64; 3];
    for p in xyz {
        let row = [p[0], p[1], 1.0];
        for i in 0..3 {
            atz[i] += row[i] * p[2];
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    solve3(&ata, &atz)
}

/// Evaluate `Z = aX + bY + c` on an `nx × ny` grid (row-major, X=i, Y=j).
///
/// Port of `Horizons.eval_plane`.
pub fn eval_plane(nx: usize, ny: usize, a: f64, b: f64, c: f64) -> Vec<f64> {
    let mut z = vec![0.0; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            z[i * ny + j] = a * (i as f64) + b * (j as f64) + c;
        }
    }
    z
}

/// Rotate `(x, y)` about the origin by `angle_in_degrees`.
///
/// Port of `Horizons.rotate_point`.
pub fn rotate_point(x: f64, y: f64, angle_in_degrees: f64) -> (f64, f64) {
    let angle = angle_in_degrees * std::f64::consts::PI / 180.0;
    let x1 = angle.cos() * x + angle.sin() * y;
    let y1 = -angle.sin() * x + angle.cos() * y;
    (x1, y1)
}

/// Enforce non-negative layer thicknesses in a horizon stack.
///
/// Port of the clip loop in `Horizons.insert_feature_into_horizon_stack`:
/// for each deeper→shallower pair (index `i` down to `2`), if
/// `maps[..., i] - maps[..., i-1] < 0`, clip thickness to ≥0 and pull the
/// shallower horizon down onto the deeper one.
///
/// `maps` is row-major `(ni, nj, nh)`; `shape = [ni, nj, nh]`.
pub fn enforce_nonnegative_thicknesses(maps: &mut [f64], shape: [usize; 3]) {
    let [ni, nj, nh] = shape;
    assert_eq!(maps.len(), ni * nj * nh);
    if nh < 3 {
        return;
    }
    // Python: for i in range(nh - 1, 1, -1)  →  i = nh-1, nh-2, ..., 2
    for i in (2..nh).rev() {
        let mut min_thick = f64::INFINITY;
        for n in 0..(ni * nj) {
            let deep = maps[n * nh + i];
            let shallow = maps[n * nh + (i - 1)];
            min_thick = min_thick.min(deep - shallow);
        }
        if min_thick < 0.0 {
            for n in 0..(ni * nj) {
                let deep = maps[n * nh + i];
                let shallow = maps[n * nh + (i - 1)];
                let mut thick = deep - shallow;
                if thick < 0.0 {
                    thick = 0.0;
                }
                maps[n * nh + (i - 1)] = deep - thick;
            }
        }
    }
}

/// Fill discrete layer labels between successive horizons.
///
/// For each (i,j) and horizon interval `h → h+1`, voxels in
/// `[ceil(z_h), floor(z_{h+1}))` receive label `h`. Outside intervals stay 255.
///
/// `depth_maps` is row-major `(ni, nj, nh)`; returns `(ni, nj, n_samples)`.
pub fn fill_layer_labels(depth_maps: &[f64], shape: [usize; 3], n_samples: usize) -> Vec<u8> {
    let [ni, nj, nh] = shape;
    assert_eq!(depth_maps.len(), ni * nj * nh);
    let mut labels = vec![255u8; ni * nj * n_samples];
    for i in 0..ni {
        for j in 0..nj {
            for h in 0..(nh.saturating_sub(1)) {
                let z0_raw = depth_maps[(i * nj + j) * nh + h];
                let z1_raw = depth_maps[(i * nj + j) * nh + h + 1];
                let z0 = (z0_raw.ceil() as isize).clamp(0, n_samples as isize) as usize;
                let z1 = (z1_raw.floor() as isize).clamp(0, n_samples as isize) as usize;
                if z1 > z0 {
                    let base = (i * nj + j) * n_samples;
                    for z in z0..z1 {
                        labels[base + z] = h as u8;
                    }
                }
            }
        }
    }
    labels
}

fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    // Gaussian elimination with partial pivoting.
    let mut m = [[0.0_f64; 4]; 3];
    for i in 0..3 {
        m[i][0] = a[i][0];
        m[i][1] = a[i][1];
        m[i][2] = a[i][2];
        m[i][3] = b[i];
    }
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if piv != col {
            m.swap(piv, col);
        }
        let diag = m[col][col];
        assert!(
            diag.abs() > 1e-15,
            "singular design matrix in fit_plane_lsq"
        );
        for c in col..4 {
            m[col][c] /= diag;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            for c in col..4 {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}
