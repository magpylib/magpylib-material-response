//! One source's parameters given once, (3,) and (1,), equal the same given
//! per observer row, past two chunks of observers (`magpylib.func`'s
//! broadcast, which the reference files' per-row checks do not reach).

use mmr_kernels as k;

const M: usize = 130;

fn observers() -> Vec<f64> {
    (0..3 * M)
        .map(|i| 0.003 * ((i as f64) * 0.618).sin() + 0.001 * (i % 3) as f64)
        .collect()
}

fn tiled(row: &[f64]) -> Vec<f64> {
    (0..M).flat_map(|_| row.iter().copied()).collect()
}

#[test]
fn shared_parameters_equal_rows() {
    let x = observers();
    let (d, p) = ([0.004, 0.002, 0.003], [0.3, -0.5, 1.2]);
    let (a, b, c) = ([0.001, -0.002, 0.0005], [0.004, 0.003, -0.001], [-2.5]);
    for field in [b'B', b'H'] {
        let (mut once, mut rows) = (vec![0.0; 3 * M], vec![0.0; 3 * M]);
        k::cuboid_field(field, &x, &d, &p, &mut once, false);
        k::cuboid_field(field, &x, &tiled(&d), &tiled(&p), &mut rows, false);
        assert_eq!(once, rows, "cuboid {}", field as char);
        k::polyline_field(field, &x, &a, &b, &c, &mut once, false);
        k::polyline_field(
            field,
            &x,
            &tiled(&a),
            &tiled(&b),
            &tiled(&c),
            &mut rows,
            false,
        );
        assert_eq!(once, rows, "polyline {}", field as char);
        k::sphere_field(field, &x, &[0.004], &p, &mut once, false);
        k::sphere_field(field, &x, &[0.004; M], &tiled(&p), &mut rows, false);
        assert_eq!(once, rows, "sphere {}", field as char);
        k::dipole_field(field, &x, &p, &mut once, false);
        k::dipole_field(field, &x, &tiled(&p), &mut rows, false);
        assert_eq!(once, rows, "dipole {}", field as char);
        k::circle_field(field, &x, &[0.004], &c, &mut once, false);
        k::circle_field(field, &x, &[0.004; M], &tiled(&c), &mut rows, false);
        assert_eq!(once, rows, "circle {}", field as char);
    }
}
