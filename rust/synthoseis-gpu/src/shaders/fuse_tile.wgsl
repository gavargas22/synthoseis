// Per-trace Zoeppritz PP + same-mode wavelet convolution (f32).
// Complex arithmetic mirrors synthoseis_seismic::zoeppritz_pp (num_complex).
// GPU f32 vs CPU f64 → near-parity, not bit-identical (see crate docs).
//
// Storage bindings kept to 4 for downlevel adapters (llvmpipe / CI):
//   1 labels, 2 trends_wavelet, 3 scratch, 4 tile_out (+ uniform params).

struct FuseParams {
    nj: u32,
    nk: u32,
    i0: u32,
    i1: u32,
    j0: u32,
    j1: u32,
    wavelet_len: u32,
    wavelet_off: u32, // index into trends_wavelet where wavelet samples begin
    angle_deg: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}

@group(0) @binding(0) var<uniform> params: FuseParams;
@group(0) @binding(1) var<storage, read> labels: array<u32>;
@group(0) @binding(2) var<storage, read> trends_wavelet: array<f32>; // 9*nk then wavelet
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> tile_out: array<f32>;

struct C {
    re: f32,
    im: f32,
}

fn c_add(a: C, b: C) -> C { return C(a.re + b.re, a.im + b.im); }
fn c_sub(a: C, b: C) -> C { return C(a.re - b.re, a.im - b.im); }
fn c_mul(a: C, b: C) -> C {
    return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
fn c_scale(a: C, s: f32) -> C { return C(a.re * s, a.im * s); }
fn c_div(a: C, b: C) -> C {
    let d = b.re * b.re + b.im * b.im;
    return C((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}
fn sinh_f(x: f32) -> f32 {
    let e = exp(x);
    let ie = exp(-x);
    return 0.5 * (e - ie);
}
fn cosh_f(x: f32) -> f32 {
    let e = exp(x);
    let ie = exp(-x);
    return 0.5 * (e + ie);
}
fn c_sin(z: C) -> C {
    return C(sin(z.re) * cosh_f(z.im), cos(z.re) * sinh_f(z.im));
}
fn c_cos(z: C) -> C {
    return C(cos(z.re) * cosh_f(z.im), -sin(z.re) * sinh_f(z.im));
}
fn c_sqrt(z: C) -> C {
    let x = z.re;
    let y = z.im;
    let r = sqrt(x * x + y * y);
    let t = sqrt(max(0.5 * (r + x), 0.0));
    var u: f32;
    if (y < 0.0) {
        u = -sqrt(max(0.5 * (r - x), 0.0));
    } else {
        u = sqrt(max(0.5 * (r - x), 0.0));
    }
    return C(t, u);
}
fn c_log(z: C) -> C {
    return C(0.5 * log(z.re * z.re + z.im * z.im), atan2(z.im, z.re));
}
fn c_asin(z: C) -> C {
    let z2 = c_mul(z, z);
    let one_m = c_sub(C(1.0, 0.0), z2);
    let s = c_sqrt(one_m);
    let iz = C(-z.im, z.re);
    let w = c_log(c_add(iz, s));
    return C(w.im, -w.re);
}

fn zoeppritz_pp(
    vp1: f32, vs1: f32, rho1: f32,
    vp2: f32, vs2: f32, rho2: f32,
    angle_deg: f32,
) -> f32 {
    let theta = C(radians(angle_deg), 0.0);
    let p = c_scale(c_sin(theta), 1.0 / vp1);
    let theta2 = c_asin(c_scale(p, vp2));
    let phi1 = c_asin(c_scale(p, vs1));
    let phi2 = c_asin(c_scale(p, vs2));

    let sin_phi1 = c_sin(phi1);
    let sin_phi2 = c_sin(phi2);
    let sin_phi1_sq = c_mul(sin_phi1, sin_phi1);
    let sin_phi2_sq = c_mul(sin_phi2, sin_phi2);
    let cos_theta = c_cos(theta);
    let cos_theta2 = c_cos(theta2);
    let cos_phi1 = c_cos(phi1);
    let cos_phi2 = c_cos(phi2);

    let a = c_sub(
        c_scale(c_sub(C(1.0, 0.0), c_scale(sin_phi2_sq, 2.0)), rho2),
        c_scale(c_sub(C(1.0, 0.0), c_scale(sin_phi1_sq, 2.0)), rho1),
    );
    let b = c_add(
        c_scale(c_sub(C(1.0, 0.0), c_scale(sin_phi2_sq, 2.0)), rho2),
        c_scale(sin_phi1_sq, 2.0 * rho1),
    );
    let c = c_add(
        c_scale(c_sub(C(1.0, 0.0), c_scale(sin_phi1_sq, 2.0)), rho1),
        c_scale(sin_phi2_sq, 2.0 * rho2),
    );
    let d = C(2.0 * (rho2 * vs2 * vs2 - rho1 * vs1 * vs1), 0.0);

    let e = c_add(
        c_scale(c_mul(b, cos_theta), 1.0 / vp1),
        c_scale(c_mul(c, cos_theta2), 1.0 / vp2),
    );
    let f = c_add(
        c_scale(c_mul(b, cos_phi1), 1.0 / vs1),
        c_scale(c_mul(c, cos_phi2), 1.0 / vs2),
    );
    let g = c_sub(
        a,
        c_mul(c_mul(c_scale(c_mul(d, cos_theta), 1.0 / vp1), cos_phi2), C(1.0 / vs2, 0.0)),
    );
    let h = c_sub(
        a,
        c_mul(c_mul(c_scale(c_mul(d, cos_theta2), 1.0 / vp2), cos_phi1), C(1.0 / vs1, 0.0)),
    );

    let det = c_add(c_mul(e, f), c_mul(c_mul(g, h), c_mul(p, p)));
    let term1 = c_mul(
        f,
        c_sub(
            c_scale(c_mul(b, cos_theta), 1.0 / vp1),
            c_scale(c_mul(c, cos_theta2), 1.0 / vp2),
        ),
    );
    let inner = c_add(
        a,
        c_mul(c_mul(c_scale(c_mul(d, cos_theta), 1.0 / vp1), cos_phi2), C(1.0 / vs2, 0.0)),
    );
    let term2 = c_mul(c_mul(h, c_mul(p, p)), inner);
    let zoep = c_div(c_sub(term1, term2), det);
    return zoep.re;
}

fn props(lab: u32, k: u32, nk: u32) -> vec3<f32> {
    var base: u32;
    if (lab == 0u) {
        base = 0u;
    } else if (lab == 1u) {
        base = 3u;
    } else {
        base = 6u;
    }
    let vp = trends_wavelet[base * nk + k];
    let vs = trends_wavelet[(base + 1u) * nk + k];
    let rho = trends_wavelet[(base + 2u) * nk + k];
    return vec3(vp, vs, rho);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = params.i1 - params.i0;
    let tj = params.j1 - params.j0;
    let n_traces = ti * tj;
    let t = gid.x;
    if (t >= n_traces) {
        return;
    }
    let di = t / tj;
    let dj = t - di * tj;
    let i = params.i0 + di;
    let j = params.j0 + dj;
    let nk = params.nk;
    let nj = params.nj;
    let base = t * nk;

    if (nk >= 2u) {
        for (var k: u32 = 0u; k < nk - 1u; k = k + 1u) {
            let idx0 = (i * nj + j) * nk + k;
            let idx1 = idx0 + 1u;
            let p0 = props(labels[idx0], k, nk);
            let p1 = props(labels[idx1], k + 1u, nk);
            scratch[base + k] = zoeppritz_pp(
                p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, params.angle_deg,
            );
        }
    }
    scratch[base + nk - 1u] = 0.0;

    let wlen = params.wavelet_len;
    let woff = params.wavelet_off;
    if (wlen == 0u) {
        for (var k: u32 = 0u; k < nk; k = k + 1u) {
            tile_out[base + k] = scratch[base + k];
        }
        return;
    }
    let start = (wlen - 1u) / 2u;
    for (var k: u32 = 0u; k < nk; k = k + 1u) {
        let n = start + k;
        var acc: f32 = 0.0;
        for (var i_s: u32 = 0u; i_s < nk; i_s = i_s + 1u) {
            if (n >= i_s) {
                let j_k = n - i_s;
                if (j_k < wlen) {
                    acc = acc + scratch[base + i_s] * trends_wavelet[woff + j_k];
                }
            }
        }
        tile_out[base + k] = acc;
    }
}
