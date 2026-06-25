# BEDLAM360 v3 Freeze Notes

## Current Goal
Freeze the BEDLAM360 v3 pipeline in a known-good state, stop active debugging of the beige/grey human problem, and preserve the lessons learned so future sessions can restart quickly with less token overhead.

## What Currently Works
- v3 planning and selection are stable enough for seed_101 with the final spatial ERP + dedup selection.
- The canonical RGB fast path works in the known reliable manual workflow when Unreal is already initialized and the USD scene is loaded.
- The render pipeline can produce RGB PNGs, preview MP4s, GT, and bridge reports when launched in the stable manual path.
- The frozen lighting policy keeps house-wide lights enabled during 360 renders; it does not gate illumination to only the selected mini-scene room.
- The default dataset lighting now adds a global soft house fill layer so adjacent rooms stay readable in ERP views; the default fill intensity was tuned down a bit to avoid washing out the scene.
- The seed_101 v3 manifest was successfully reduced to a compact 20-scene selection with improved motion/identity diversity.

## Known Failure
- Cold or safe Unreal launches can produce beige/grey humans and sometimes beige indoor materials.
- The failure is environment/session dependent, not a camera/export-only issue.
- The most reliable manual workaround was:
  1. Open Unreal manually.
  2. Load the prebuilt seed_101 map.
  3. Run the official renderer once.
  4. Run the exact same renderer again in the same Unreal session.
  5. The second run looks correct.

## Conditions Where the Issue Appeared
- Fresh Unreal sessions launched from Linux.
- Automated launches that did not preserve the same editor state as the manual session.
- Bootstrap attempts that did not actually open the same editor map or did not reach the same state as the manual workflow.
- Some runner attempts that used warm-up/probe logic but still did not mimic the manual first real render.

## Conditions Where It Improved or Disappeared
- Manual Unreal session with the preloaded `BEDLAM360_seed101_loaded` map.
- Second render in the same Unreal session after one sacrificial real render.
- Direct renderer launch from within an already-open, already-loaded Unreal editor session.

## Tests Already Performed
- Batch rendering with `dataset_rgb_fast` and `ldr_passthrough`.
- `sequence_adaptive_rgb` experiments.
- Direct PNG export path experiments.
- Cubemap artifact diagnostics and retry experiments.
- Spatial ERP selection and spatial deduplication experiments.
- Planner-vs-runtime trajectory consistency audits.
- Human-scene collision audits.
- Asset registry and clothing/texture path resolution audits.
- Preloaded editor map opening and validation attempts.
- Bootstrap launcher experiments from Linux.
- Appearance warm-up experiments.
- Resource monitor and VRAM accumulation logging.

## Bootstrap Launcher Experiment
- A separate bootstrap launcher was created to wait for editor/asset/USD readiness using tick callbacks.
- It evolved into a safer async launcher path, but it still depended on opening the right editor map and matching the manual editor state.
- The bootstrap path is experimental and should not be treated as the minimal operational recipe unless explicitly needed.

## Warm-up Experiments
- Tiny 0..1 warm-ups were insufficient.
- Longer warm-ups helped only when they were close to a real render path.
- A sacrificial first render in the same session matched the manual workaround better than sleep-based waiting.
- The most useful warm-up findings suggest the first complete render triggers initialization that later renders reuse.

## Asset / VRAM / GPU Observations
- GPU memory rises clip after clip inside a single Unreal session.
- CPU RSS can partially fall after cleanup, but GPU memory often stays high.
- Stable actor/component counts do not necessarily imply stable VRAM.
- SceneCapture and render target counts appeared stable in the investigated runs.
- Some debug inventories suggested that loaded assets can remain resident across clips even when gameplay actors are cleaned up.

## Hypotheses Considered
- Cold-start editor/material initialization issue.
- USD stage loading / asset registry readiness issue.
- Preferred editor map mismatch.
- Bootstrap launch timing issue.
- Warm-up not matching the manual workflow.
- Texture/material resolution issue for BEDLAM humans.
- VRAM accumulation due to retained assets, textures, or render resources.

## Hypotheses Weakened or Rejected
- SceneCaptureCube / PNG export as the sole cause of beige humans.
- Camera and tonemap settings as the only cause of the beige problem.
- A simple frame 0..1 warm-up as sufficient to match the manual workflow.
- A pure collision or planning issue as the explanation for beige humans.

## Most Plausible Remaining Hypotheses
- The automated editor bootstrap still does not exactly reproduce the manual editor state.
- The first full render in a fresh session is still the key initialization event.
- GPU memory growth is likely due to session-local asset/resource residency rather than the selection planner.

## Useful Commands
- Manual stable render path:
  - open Unreal manually
  - load `/Game/BEDLAM360_seed101_loaded.BEDLAM360_seed101_loaded`
  - run the official renderer command once
  - rerun the same command in the same Unreal session
- Direct renderer example:
  - `py /media/mathis/PANO/BEDLAM2/projects/BEDLAM360/Content/Python/render_selected_infinigen_bedlam_erp.py --scene-root ... --manifest ... --batch --batch-balanced-rooms --max-clips 1 --frame-start 0 --frame-end 120 --render-output-profile dataset_rgb_fast --rgb-tonemap-mode ldr_passthrough`

## Commands / Paths That Did Not Help
- Very short warm-ups.
- Sleep-only waiting without a real render.
- Bootstrap attempts that did not open the same preloaded map as the manual session.
- Bootstrap paths that did not reach the same material-ready editor state as the manual workflow.

## Recommended Future Diagnostic Path
1. Revisit the bootstrap only if the exact manual editor state must be automated.
2. If debugging resumes, start from the known stable manual workflow and compare only one variable at a time.
3. If VRAM leaks are the next priority, profile one subsystem at a time with a minimal appearance mode and memory checkpoints.
4. Avoid expanding the renderer until a single reproducible failure mode is isolated.

## Freeze Hygiene
- Minimal v3 rendering is preserved in the active renderer scripts; the long diagnostic detours are archived separately.
- The script inventory lives in [docs/bedlam360_python_script_inventory.md](bedlam360_python_script_inventory.md).
- The experimental archive lives in [experimental_v3_debug_archive/README.md](../experimental_v3_debug_archive/README.md).
- Current frozen defaults keep global house lighting enabled and disable `r.HairStrands.Voxelization` by default.
