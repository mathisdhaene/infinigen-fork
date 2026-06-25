# BEDLAM360 Infinigen Bridge Status

BEDLAM360 v3 should be understood as:

- BEDLAM360 v2 core renderer, ERP, GT, asset playback, and appearance pipeline
- plus the Infinigen indoor-environment bridge layer

The bridge layer adds:

- Infinigen scene metadata
- room/object collision metadata
- spawn placement
- starterpack-only manifest filtering
- mini-scene selection
- indoor light conversion
- room-aware static camera placement

## 1. Current validated pipeline

- Current phase is `starterpack-only`.
- Full BEDLAM training-motion coverage is deferred until the house-scale pipeline is stable.
- BEDLAM360 v2 remains the rendering foundation:
  - canonical BEDLAM spawn path
  - GeometryCache playback
  - clothes / hair / shoes / textures
  - ERP rendering path
  - GT export / manifest conventions
  - natural-timing / LevelSequence timing fixes
- The Infinigen bridge extends that existing v2 pipeline rather than replacing it.

- Infinigen indoor scenes are exported to USD and loaded in Unreal through a `UsdStageActor`.
- Human placement uses the validated `T_infinigen_to_bedlam` bridge:
  - `X_bedlam_cm = 100 * X_infinigen_m`
  - `Y_bedlam_cm = -100 * Y_infinigen_m`
  - `Z_bedlam_cm = 100 * Z_infinigen_m + SCENE_FLOOR_OFFSET_CM`
  - `yaw_bedlam_deg = -degrees(yaw_infinigen_rad)`
- `SCENE_FLOOR_OFFSET_CM = 14.0` is part of the active bridge and is required for visible floor alignment.
- Validated spawn placement works for offline-selected mini-scenes and does not rely on ad-hoc runtime body offsets.
- BEDLAM appearance path is active:
  - body texture
  - clothing when available
  - hair when available
  - shoes when available
- ERP capture source mode is currently:
  - `ERP_CAPTURE_SOURCE_MODE = "ldr_final_color"`

Output contract:

- `*_rgb.png` = BEDLAM-like Unreal `FinalColor` LDR RGB
- `*_erp.exr` = HDR/debug output
- `preview_rgb.mp4` = RGB sequence preview
- `preview_adaptive.mp4` = adaptive-preview sequence preview
- adaptive/fixed preview PNGs are debug-only and are not canonical dataset RGB

## 2. Lighting policy

Current lighting policy is intentionally narrow and frozen around one primary bridge path.

- `LIGHTING_ENVIRONMENT_MODE = "indoor_balanced"`
  - current frozen default for v3 dataset rendering
  - keep converted indoor practicals / fallback fill
  - reintroduce subdued BEDLAM2-style HDRI / skylight / outdoor contribution
- Supported environment modes:
  - `"indoor_isolated"`
    - suppress bright exterior/background contribution for hard indoor debugging
  - `"indoor_balanced"`
    - current default
    - use subdued outdoor/HDRI contribution plus indoor practical/fallback lights
  - `"bedlam2_default"`
    - restore BEDLAM2-like environment contribution as closely as possible
- `ENV_LIGHT_SCALE = 0.3`
  - current subdued skylight/environment contribution for `indoor_balanced`
- `BALANCED_DIRECTIONAL_LIGHT_SCALE = 0.25`
  - current subdued sun/directional contribution for `indoor_balanced`
- BEDLAM2 HDRI support is reused through existing BEDLAM360 v2 helpers when available.

Indoor practical lighting:

- Native Infinigen practical lights are not trusted through Unreal USD import directly, so the bridge converts selected USD/Blender light information into native Unreal lights.
- Blender light manifest is used because USD practical-light intensity was effectively flattened to `1.0`, which is not physically meaningful.
- Lumens to candela conversion is explicit for converted Unreal `PointLight` actors:
  - `candela ~= lumens / (4 * pi)`

Current key knobs:

- `USD_POINT_LIGHT_BASE_LUMENS = 200.0`
- `BLENDER_POINT_LIGHT_REFERENCE_ENERGY = 50.0`
- `CONVERTED_CEILING_LIGHT_Z_OFFSET_CM = -50.0`
- `CONVERTED_CEILING_LIGHT_MIN_WALL_CLEARANCE_M = 0.25`
- `CONVERTED_CEILING_LIGHT_INWARD_OFFSET_M = 0.20`
- `CONVERTED_POINT_LIGHT_ATTENUATION_RADIUS_CM = 500.0`
- `CONVERTED_POINT_LIGHT_SOURCE_RADIUS_CM = 20.0`
- `CONVERTED_POINT_LIGHT_USE_INVERSE_SQUARED_FALLOFF = True`

Fallback fill policy:

- use converted USD/Blender `SphereLight` practicals first
- if the selected room has no nearby practical light, add exactly one soft fallback `PointLight`
- fallback fill placement is now room/obstacle/human-aware:
  - room polygon clearance
  - obstacle-footprint clearance
  - human-anchor clearance
  - candidate scoring instead of fixed room-centroid placement

USD light-type policy:

- USD `RectLight` prims are ignored by default because their transforms were unreliable after import
- USD `SphereLight` practicals remain the main converted-light path

What was debugged and rejected:

- old generic room `RectLight` planner
- old debug key/fill rig behavior
- stale `SceneCaptureCube` post-process exposure override experiments
- treating adaptive preview PNGs as canonical RGB
- treating Unreal `PointLight` intensities as if they were direct lumens instead of candelas

## 3. Cleanup / freeze summary

The bridge now distinguishes between the frozen v3 path and optional retained diagnostics.

KEEP:

- starterpack-only manifest support
- strict manifest/miniscene validation
- scene-root rebinding
- USD stage consistency check
- Blender light-manifest driven `SphereLight -> PointLight` conversion
- room-aware fallback fill placement
- camera obstacle/human/wall-aware placement
- frame-range clamping to motion-supported range
- `*_rgb.png`, `*_erp.exr`
- `preview_rgb.mp4`, `preview_adaptive.mp4`
- useful final report summaries

KEEP_DEBUG:

- `CAPTURE_PIPELINE_DIAGNOSTICS`
  - now default `False`
- `LIGHTING_INTENSITY_SWEEP`
  - retained as an explicit debug tool
- fixture / `CeilingLightFactory` diagnostics
  - retained but disabled by default through `ENABLE_FIXTURE_DIAGNOSTICS = False`
- manual appearance warmup helper:
  - `/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/warmup_selected_infinigen_bedlam_appearance.py`

REMOVE or disabled from active v3 path:

- automatic in-render appearance warmup path in `render_selected_infinigen_bedlam_erp.py`
  - replaced by the manual warmup script
- old generic room `RectLight` planner
- old debug key/fill rig behavior
- always-on fixture deep-diagnostic path
  - now explicit debug-only instead of part of every v3 run
- exposure-override experiments in the active render path

## 4. Current frozen defaults

- BEDLAM360 v3 = BEDLAM360 v2 renderer / GT / ERP + Infinigen indoor bridge
- current default environment mode:
  - `LIGHTING_ENVIRONMENT_MODE = "indoor_balanced"`
- current RGB output convention:
  - `ERP_CAPTURE_SOURCE_MODE = "ldr_final_color"`
- current floor alignment:
  - `SCENE_FLOOR_OFFSET_CM = 14.0`
- current diagnostics policy:
  - keep diagnostics available
  - keep them off by default for dataset rendering

## 5. Known unresolved issues

- Multi-human scenes can still look cloned or synchronized.
- The motion phase-offset patch may not yet be affecting the actually rendered `GeometryCache` playback as intended.
- We still need runtime verification of the evaluated frame per human during actual render.
- Visible ceiling fixture meshes are still missing in Unreal.
  - Blender and USD contain the expected `CeilingLightFactory_*` fixtures
  - practical lights are converted successfully
  - visible fixture geometry is not currently exposed as normal Unreal meshes/components
- Some non-kitchen rooms remain empty, coarse, or weakly lit because the original generation and sampling setup was limited/coarse.

## 6. Next stage: full sequence rendering

Target:

- render full AMASS/BEDLAM motions
- use selected mini-scenes from the manifest
- support a moving ERP camera inside the room
- camera height should be planned around `1.20 m` above the visible Infinigen house floor
- remember that the visible house floor is offset relative to Unreal floor:
  - `SCENE_FLOOR_OFFSET_CM = 14.0`

## 7. Camera requirements

Planned camera support should remain room-aware.

- define camera positions in Infinigen coordinates first
- convert camera poses with the same `T_infinigen_to_bedlam` logic
- target camera height:
  - `camera_z_m = room_floor_z_m + 1.20`
  - then convert to Unreal and apply the same floor-offset handling
- support modes:
  - static camera
  - slow moving camera
  - close humans
  - very close humans
  - humans near ERP seam
  - humans partially cut at ERP edge

## 8. Immediate next bugfix

Priority should stay on multi-human cloning/synchronization, not new lighting features.

Immediate checks:

- verify actual rendered `GeometryCache` section offsets
- verify actual evaluated frame per human at render time
- verify actor labels and sequence bindings are unique per human
- verify different body/appearance presets are truly used at runtime
- if the same motion is used twice:
  - either reject that pair during selection
  - or apply a proven runtime phase offset confirmed in the rendered result

## 9. Files of interest

- Infinigen planning / metadata:
  - `outputs/indoors/human_spawn_poc/scene_collision_metadata.json`
  - `outputs/indoors/human_spawn_poc/human_spawn_poses.json`
  - `outputs/indoors/human_spawn_poc/miniscene_selection_v0/bedlam360_infinigen_miniscenes.json`
  - `outputs/indoors/human_spawn_poc/miniscene_selection_v0/bedlam360_infinigen_miniscenes_starterpack_only.json`
  - `infinigen_examples/prepare_bedlam360_v3_scene_batch.py`
  - `infinigen_examples/validate_bedlam360_v3_acceptance.py`
- Unreal bridge scripts:
  - `/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_validated_infinigen_bedlam_erp.py`
  - `/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py`
  - `/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/warmup_selected_infinigen_bedlam_appearance.py`
- Preview/debug tools:
  - `/media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Python/bedlam360_preview_tools.py`

## 10. Practical batch prep entry point

For the current starterpack-only phase, scene preparation should start from:

- `infinigen_examples/prepare_bedlam360_v3_scene_batch.py`

This script is intended to:

- generate multiple complete indoor scenes with different seeds
- export USD for each scene
- export:
  - `scene_collision_metadata.json`
  - `human_spawn_poses.json`
  - `solve_state.json`
  - `blender_light_manifest.json`
  - `usd_light_manifest.json`
- run the starterpack-only mini-scene planner
- write:
  - `bedlam360_infinigen_miniscenes_starterpack_only.json`
  - `starterpack_manifest_filter_report.json`
  - a shell recipe plus a batch-prep JSON report

## 11. V3 acceptance validation

Before starting v4 robot-camera trajectories, v3 should be checked with:

- `infinigen_examples/validate_bedlam360_v3_acceptance.py`

This acceptance script validates, per generated scene:

- required bridge artifacts exist
- starterpack-only manifest structure is valid
- at least one single-human mini-scene exists
- at least one two-human mini-scene exists when the filtered planner report says it should
- two-human mini-scenes do not contain duplicate motion IDs
- all motion IDs remain within the starterpack whitelist

Optional smoke-test mode checks the existing Unreal selected-miniscene renderer and is intended to answer one question cleanly:

- `ready_for_v4 = true/false`
