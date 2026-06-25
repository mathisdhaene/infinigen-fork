# BEDLAM360 v3 Operational Recipe

This is the smallest known reliable way to run BEDLAM360 v3 without re-entering the beige/grey debug spiral.

## Recommended Manual Workflow
1. Launch Unreal Editor manually.
2. Open the preloaded seed_101 map:
   - `/Game/BEDLAM360_seed101_loaded.BEDLAM360_seed101_loaded`
3. Keep the editor session alive.
4. Run the official renderer command from the Unreal Python console.
5. If the first render looks wrong, run the exact same command a second time in the same Unreal session.

## Canonical Command
```bash
py "/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py" \
  --scene-root "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101" \
  --manifest "/media/mathis/PANO/infinigen/outputs/indoors/bedlam360_v3_scenes/seed_101/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json" \
  --batch --batch-balanced-rooms --max-clips 1 \
  --frame-start 0 --frame-end 120 \
  --render-output-profile dataset_rgb_fast \
  --rgb-tonemap-mode ldr_passthrough
```

## Reliable Variants
- For a smoke test, reduce `--frame-end` to `20` or `30`.
- Keep `--render-output-profile dataset_rgb_fast` and `--rgb-tonemap-mode ldr_passthrough` for the fast canonical RGB path.
- The renderer now keeps house-wide lighting enabled for 360 renders; it does not restrict lights to only the selected mini-scene room.
- Global soft house fill is enabled by default. Use `--disable-global-house-fill` only if you need to compare against the older darker lighting policy. The default fill intensity is intentionally modest after the latest tuning.
- Use the same exact command twice in the same Unreal session if the first render is beige.

## Not Part Of The Minimal Recipe
- Bootstrap launcher experiments.
- Safe launcher / resilient runner experiments.
- Resource monitor / memreport / RHI dump profiling.
- Appearance debug modes like `no_humans`, `body_only`, `body_no_hair`, `body_no_clothing`.
- Warm-up-only scripts that do not reproduce the manual two-render behavior.

## Expected Outputs
- RGB PNG sequence.
- Preview MP4.
- Bridge report JSON.
- Frame validity index.
- Any configured memory checkpoints only if explicitly enabled.

## What To Keep Stable
- Scene root.
- Manifest path.
- Preferred preloaded map.
- Camera placement.
- Lighting setup.
- Output profile.
- Tonemap mode.

## Frozen Defaults
- Global house fill lighting is enabled by default for 360 readability.
- `r.HairStrands.Voxelization 0` remains the default for stability.
- The minimal supported path is still the direct renderer command from a manually prepared Unreal session.
- Large batch automation remains experimental and is not the primary supported operational path yet.
