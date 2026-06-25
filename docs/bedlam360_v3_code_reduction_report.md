# BEDLAM360 v3 Code Reduction Report

Date: 2026-06-24

## Scope

This pass was a cleanup-only reduction on:

`/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py`

The goal was to keep the current minimal render path intact while removing dead code and freezing experimental diagnostics out of the active renderer.

## Line Count

- Original line count: `5101`
- Final line count: `4679`
- Net reduction: `422` lines

## Cleanup Plan Used

Classification applied before editing:

- `REQUIRED_FOR_MINIMAL_RENDER`
  - `render_selected_infinigen_bedlam_erp()`
  - batch selection
  - scene/USD binding
  - `render_full_appearance_sequence_to_root(...)`
  - output writing
  - cleanup path

- `OPTIONAL_DEBUG`
  - asset residency / memory checkpoint plumbing
  - cleanup summaries
  - runtime human verification
  - preview generation
  - capture diagnostics

- `EXPERIMENTAL`
  - renderability reports
  - motion listing reports
  - offline trajectory audit
  - lighting sweep mode
  - multi-human runtime debug report

- `OBSOLETE`
  - unused helper functions with zero references

- `DUPLICATED`
  - none removed in this pass

- `UNUSED`
  - `_paths_union`
  - `_load_miniscene`

## Removed From Active Implementation

These functions were removed entirely because they had no references:

- `_paths_union`
- `_load_miniscene`

These experimental blocks were frozen out of the active renderer and replaced with short archived stubs:

- `_list_renderable_miniscenes_report`
- `_diagnose_renderability_report`
- `_list_available_motions_report`
- `_offline_run_trajectory_consistency_audit`
- `_multi_human_runtime_debug`
- `_render_lighting_intensity_sweep`

Why this was safe:

- none of the removed helpers are required for the minimal render command
- the experimental/report-only paths are not part of the frozen v3 operational recipe
- the replacement stubs fail loudly or return a compact archived summary instead of silently changing render output

## Archived Scripts / Files

These experimental helpers were already copied into:

`/media/mathis/PANO/infinigen/experimental_v3_debug_archive/`

Archived items:

- `render_selected_infinigen_bedlam_erp_before_cleanup.py`
- `bootstrap_bedlam360_render.py`
- `run_bedlam360_v3_resilient_render.py`
- `audit_bedlam360_v3_diversity.py`
- `prepare_bedlam360_v3_scene_batch.py`
- `spawn_debug_human_spawn.py`
- `spawn_validated_bedlam_motion.py`
- `warmup_selected_infinigen_bedlam_appearance.py`
- `inspect_ceiling_light_transforms.py`
- `validate_bedlam360_v3_acceptance.py`
- `validate_scene_collision_metadata.py`

Archive README:

- `experimental_v3_debug_archive/README.md`

## Preserved Flags

Kept for compatibility with the frozen operational path:

- `--scene-root`
- `--manifest` / `--manifest-path`
- `--miniscene-id`
- `--miniscene-room`
- `--miniscene-index`
- `--batch`
- `--max-clips`
- `--batch-room-filter`
- `--batch-balanced-rooms`
- `--frame-start`
- `--frame-end`
- `--render-output-profile`
- `--rgb-tonemap-mode`
- `--reject-clips-with-artifacts`
- `--pause-after-spawn-before-render`
- `--pause-after-spawn-seconds`
- `--render-warmup-frame-count`
- `--discard-warmup-frames`
- `--probe-frame-before-render`
- `--reject-beige-probe`
- `--bedlam-debug-appearance-mode`
- `--emit-memreport`
- `--emit-rhi-memory-dump`

## Deprecated / Archived Flags

These remain frozen out of the active implementation and should be treated as archived debug-only paths:

- `--list-renderable`
- `--diagnose-renderability`
- `--list-available-motions`
- `--offline-run-root`
- `--lighting-intensity-sweep`

## Minimal Command Preserved

The minimal known-good direct render command remains:

```bash
py "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py" \
  --scene-root "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101" \
  --manifest "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json" \
  --batch --batch-balanced-rooms --max-clips 1 \
  --frame-start 0 --frame-end 120 \
  --render-output-profile dataset_rgb_fast \
  --rgb-tonemap-mode ldr_passthrough
```

## Validation

Syntax check run successfully:

```bash
python3 -m py_compile /media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py
```

## Notes

What was intentionally not touched:

- camera math
- USD scene selection logic
- BEDLAM actor setup
- render output layout
- minimal clip rendering path

