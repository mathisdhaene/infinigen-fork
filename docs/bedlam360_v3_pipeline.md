# BEDLAM360 v3 Pipeline

This is the minimal end-to-end BEDLAM360 v3 workflow, in order.

## 1. Generate the Infinigen scene

Use the batch prep script to build one or more `seed_*` scene roots, export the USD scene, export light manifests, and plan the starterpack-only mini-scenes.

```bash
cd /media/mathis/PANO/infinigen
python3 infinigen_examples/prepare_bedlam360_v3_scene_batch.py \
  --output-root outputs/indoors/bedlam360_v3_scenes \
  --seed-start 101 \
  --num-scenes 3 \
  --frame-start 0 \
  --frame-end 120
```

What this produces per seed:
- `scene.blend`
- `scene_collision_metadata.json`
- `human_spawn_poses.json`
- `usd_export/export_scene.blend/export_scene.usdc`
- `miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json`
- `miniscene_selection_v0/starterpack_manifest_filter_report.json`
- `miniscene_selection_v0/starterpack_render_recipe.json`

If you want to run the low-level steps manually instead of the wrapper, the order is:
1. `infinigen_examples/generate_indoors.py`
2. `infinigen.tools.export`
3. `infinigen_examples/export_blender_light_manifest.py`
4. `infinigen_examples/export_usd_light_manifest.py`
5. `infinigen_examples/plan_bedlam360_miniscenes.py`

## 2. Build the human manifest

The manifest is already produced by the prep script above, but the direct planner command is:

```bash
cd /media/mathis/PANO/infinigen
python3 infinigen_examples/plan_bedlam360_miniscenes.py \
  --metadata outputs/indoors/bedlam360_v3_scenes/seed_101/scene_collision_metadata.json \
  --motion-root-dir outputs/indoors/human_spawn_poc/motion_roots_10 \
  --output-dir outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0 \
  --motion-set-mode starterpack_only \
  --starterpack-whitelist /media/mathis/PANO/bedlam2_render/config/whitelist_animations_starterpack.json \
  --min-human-count 2 \
  --frame-start 0 \
  --frame-end 120
```

The final manifest used by Unreal is:
- `outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json`

## 3. Render in Unreal

Launch Unreal Editor, open the preloaded BEDLAM360 map if you use the manual safe workflow, then run the frozen renderer command from the Unreal Python console:

```text
py "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py" --scene-root "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101" --manifest "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json" --batch --batch-balanced-rooms --max-clips 1 --frame-start 0 --frame-end 120 --render-output-profile dataset_rgb_fast --rgb-tonemap-mode ldr_passthrough
```

For multiple scenes, increase `--max-clips` or pass a specific `--miniscene-id` list/selection strategy if you are not using the default batch picker.

### Debug GT overlay
If you want to verify 3D human GT alignment directly in the frozen v3 renderer, add:

```text
--debug-gt-overlay
```

This uses the post-render GT export plus `bedlam360_gt_erp_alignment.py` to generate joint/vertex overlays for every rendered frame under the benchmark output:

```text
exports/bedlam360_infinigen_bridge_benchmark/<run_id>/frames_*/projections2d/v3_debug_gt_overlay/overlays/
exports/bedlam360_infinigen_bridge_benchmark/<run_id>/frames_*/projections2d/v3_debug_gt_overlay/gt_overlay.mp4
```

For a quick debug sample only, add `--debug-gt-overlay-frames 12,60,104`.

For batch runs, the most reliable automatic path is to run the Linux post-step after Unreal finishes. It completes the benchmark GT export if needed and writes overlays for the most recent `N` render runs:

```bash
python3 /media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Python/bedlam360_export_gt_overlays_for_runs.py --latest 6
```

This writes both PNG overlays and `gt_overlay.mp4` by default. Use `--overlay-mp4-fps 12` to tune playback speed, or `--disable-overlay-mp4` if you only want PNGs.

To regenerate existing overlays, add `--force`.

## 4. Final outputs

The frozen renderer writes:
- RGB PNG frames
- `preview/preview_rgb.mp4`
- `bridge_report.json`
- `frame_validity_index.json` and `frame_validity_index.csv`
- benchmark GT export artifacts under the bridge output root

## 5. GT export note

The current pipeline does export human 3D ground truth, but not as a single dedicated aligned CSV.

Authoritative 3D GT is written as:
- `joints3d/joints3d.npz`
- `vertices/vertices.npz`
- `smplx/parameters.npz`

The CSV outputs are primarily metadata / evaluation helpers:
- `metadata/frame_validity_index.csv`
- `metadata/<sequence>_<range_tag>_body_evaluation.csv`
- `metadata/frame_mapping.json`

If you need ERP-aligned 2D/3D projection exports or QA plots, run:

```bash
python3 /media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Python/bedlam360_gt_erp_alignment.py \
  /media/mathis/PANO/BEDLAM2/projects/BEDLAM360/exports/bedlam360_benchmark_v0/<run_id>/frames_0012_0119
```

## 6. Minimal stable recipe

The frozen operational path is still:
1. prepare the scene and manifest
2. open Unreal
3. run the renderer command above

For the most reliable v3 behavior, keep:
- global house lighting enabled
- `r.HairStrands.Voxelization = 0`
- `render_output_profile = dataset_rgb_fast`
- `rgb_tonemap_mode = ldr_passthrough`
