/// Run chunked e2e: fused generate → optional MDIO (sub-volume chunks) → parity.
pub fn run_e2e_chunked(cfg: &E2eConfig) -> Result<(E2eReport, WorkingSetStats), String> {
    SeismicFilters::from_config(cfg)?;
    let (volumes, stats) = generate_chunked(cfg);
    let (second, _) = generate_chunked(cfg);
    let parity = parity::compare_volumes(
        &volumes.labels,
        &second.labels,
        &volumes.angle_stack,
        &second.angle_stack,
    );
    if !parity.passes_defaults() {
        return Err(format!(
            "chunked e2e self-parity failed: iou={:.6} agr={:.6} mae={:.6e} maxabs={:.6e}",
            parity.label_iou, parity.label_agreement, parity.angle_mae, parity.angle_max_abs
        ));
    }

    let mut store_path = None;
    if let Some(ref path) = cfg.store_path {
        write_e2e_mdio_chunked(path, cfg, &volumes)?;
        let opened = MdioStore::open(path).map_err(|e| e.to_string())?;
        let back_angles = opened.read_volume().map_err(|e| e.to_string())?;
        let back_labels = opened.read_labels_u8().map_err(|e| e.to_string())?;
        let mdio_parity = parity::compare_volumes(
            &volumes.labels,
            &back_labels,
            &volumes.angle_stack,
            &back_angles,
        );
        if !mdio_parity.passes_defaults() {
            return Err(format!(
                "chunked e2e MDIO round-trip parity failed: {mdio_parity:?}"
            ));
        }
        store_path = Some(path.clone());
    }

    Ok((
        E2eReport {
            volumes,
            parity,
            store_path,
            status: "ok-e2e-chunked",
        },
        stats,
    ))
}
