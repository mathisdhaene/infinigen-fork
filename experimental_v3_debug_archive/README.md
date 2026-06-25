# BEDLAM360 v3 Experimental Archive

This folder mirrors the noisy v3 debugging work so future sessions can ignore it by default.

## What This Archive Is
- A local copy of the experimental scripts that helped diagnose v3 launch, warm-up, USD, appearance, collision, and memory issues.
- Not part of the minimal operational recipe.
- Safe to consult later if we resume the beige/grey or VRAM investigations.

## Archived Scripts

### `Content_Python/bootstrap_bedlam360_render.py`
- Purpose: asynchronous Unreal bootstrap launcher that waited for asset registry and USD readiness before launching the real renderer.
- Helped: yes, as a diagnostic path.
- Failed: yes, as a stable non-interactive launch path on Linux.
- Reuse later: yes, if we need to reproduce manual editor readiness more carefully.

### `Content_Python/spawn_debug_human_spawn.py`
- Purpose: isolate BEDLAM spawn/appearance issues.
- Helped: yes for low-level inspection.
- Failed: not part of the stable operational path.
- Reuse later: maybe, only for spawn debugging.

### `Content_Python/warmup_selected_infinigen_bedlam_appearance.py`
- Purpose: warm-up experiments for the human appearance state.
- Helped: partially, mainly for hypothesis testing.
- Failed: did not become the final solution.
- Reuse later: only if the exact warm-up behavior needs to be re-tested.

### `Content_Python/spawn_validated_bedlam_motion.py`
- Purpose: standalone spawn/play helper for a single validated BEDLAM motion.
- Helped: yes, for isolated spawn and appearance checks.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if a focused spawn repro is needed again.

### `Python/debug_bedlam360_frame_export.py`
- Purpose: frame-export debug harness for BEDLAM/Capture/GeometryCache paths.
- Helped: yes during export troubleshooting.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if export regressions need a repro.

### `Python/debug_bedlam_sequence_alignment.py`
- Purpose: sequence alignment debug harness.
- Helped: yes for alignment checks.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, for regression repros.

### `Python/debug_body_facing_calibration.py`
- Purpose: body-facing calibration experiment.
- Helped: yes during calibration work.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, only for calibration reruns.

### `Python/debug_erp_axes.py`
- Purpose: ERP axes calibration/debug experiment.
- Helped: yes for axis validation.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, for ERP convention checks.

### `Python/debug_erp_camera_rotation_effect.py`
- Purpose: inspect ERP camera rotation behavior.
- Helped: yes for rotation-effect testing.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, for ERP regression checks.

### `Python/debug_geometrycache_levelsequence_validation.py`
- Purpose: validate GeometryCache playback through LevelSequence.
- Helped: yes for playback debugging.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if playback regressions return.

### `Python/debug_geometrycache_motion_scan.py`
- Purpose: scan GeometryCache motion behavior.
- Helped: yes for motion scan troubleshooting.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, for future motion regressions.

### `Python/debug_geometrycache_playback_validation.py`
- Purpose: GeometryCache playback validation harness.
- Helped: yes for playback/validation checks.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if GeometryCache playback must be revalidated.

### `Python/debug_groom_runtime.py`
- Purpose: runtime groom debug helper.
- Helped: yes for grooming diagnostics.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if groom runtime bugs return.

### `Python/debug_v1_projection_alignment.py`
- Purpose: v1 projection alignment debug helper.
- Helped: yes for historical alignment checks.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if the old projection path is revisited.

### `Python/bedlam360_generate_v1_stress_test.py`
- Purpose: legacy v1 stress harness for GT/export behavior.
- Helped: yes for stress testing.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, if v1-style stress tests are needed again.

### `Python/bedlam360_erp_convention_calibration.py`
- Purpose: ERP convention calibration experiment.
- Helped: yes for convention discovery.
- Failed: it was an experiment, not the frozen path.
- Reuse later: maybe, for future calibration work.

### `Python/bedlam360_erp_convention_calibration_post.py`
- Purpose: calibration postprocess experiment.
- Helped: yes for calibration analysis.
- Failed: not part of the frozen v3 operational path.
- Reuse later: maybe, for future calibration work.

## Legacy Versions

### `legacy_versions/bedlam360_generate_v0_dataset.py`
- Purpose: legacy v0 dataset generator.
- Helped: yes historically, but it is not needed for frozen v3.
- Failed: not part of the frozen v3 operational path.
- Reuse later: only if the old v0 pipeline must be reproduced.

### `legacy_versions/bedlam360_generate_v1_dataset.py`
- Purpose: legacy v1 dataset generator.
- Helped: yes historically, but it is not needed for frozen v3.
- Failed: not part of the frozen v3 operational path.
- Reuse later: only if the old v1 pipeline must be reproduced.

### `infinigen_examples/run_bedlam360_v3_resilient_render.py`
- Purpose: chunked Unreal supervisor with retries, warm-up, bootstrap, and resource monitoring.
- Helped: yes for diagnosing crash stages and resource growth.
- Failed: too complex and not part of the simplest known-good manual recipe.
- Reuse later: yes, if we need supervised chunked rendering again.

### `infinigen_examples/audit_bedlam360_v3_diversity.py`
- Purpose: candidate diversity audit and selection analysis.
- Helped: yes.
- Failed: not a rendering path, only an analysis helper.
- Reuse later: yes, for future planner audits.

### `infinigen_examples/inspect_ceiling_light_transforms.py`
- Purpose: inspect lighting transforms and fallback fill placement.
- Helped: yes during lighting diagnosis.
- Failed: not needed for the minimal render recipe.
- Reuse later: maybe, only for lighting regressions.

### `infinigen_examples/prepare_bedlam360_v3_scene_batch.py`
- Purpose: batch preparation helper for v3 scene generation.
- Helped: yes for pipeline setup.
- Failed: not part of the minimal render recipe.
- Reuse later: yes, if batch preparation is revisited.

### `infinigen_examples/validate_bedlam360_v3_acceptance.py`
- Purpose: acceptance validation for generated scenes/manifests.
- Helped: yes.
- Failed: not needed for the minimal render recipe.
- Reuse later: yes, for acceptance audits.

### `infinigen_examples/validate_scene_collision_metadata.py`
- Purpose: collision metadata validation for planner output.
- Helped: yes.
- Failed: not needed for the minimal render recipe.
- Reuse later: yes, if collision audits resume.

### `infinigen_examples/human_spawn_sampler.py`
- Purpose: human spawn sampling experiments.
- Helped: yes during planner exploration.
- Failed: not needed for the minimal render recipe.
- Reuse later: maybe, if spawn distribution work resumes.

## Recommended Minimal Path
- Prefer the manual Unreal session with the preloaded `BEDLAM360_seed101_loaded` map.
- Use the direct renderer command from the operational recipe.
- Avoid the bootstrap and resilient-runner paths unless you are explicitly re-testing editor initialization.
