# RodSizer Parameter Guide

Use this guide to choose calibration and separation settings and interpret the resulting measurements.

## Detection flow

The backend uses an AutoDetect-mNP-style workflow in `backend/processing.py`:

1. `K-means segmentation` separates foreground particles from background.
2. `Scale bar masking` removes the scale bar area so it is not detected as a particle.
3. `Solidity-based split logic` decides whether an object is treated as a single particle (solidity > 0.9) or sent to clump splitting.
4. `Clump splitting` separates connected regions with a marker-controlled **watershed on the Euclidean distance transform** (`autodetect_utils.split_clump`). Extended (h-)maxima define the particle seeds; the separation slider adjusts the depth needed to distinguish neighboring peaks. Inspect the overlay to confirm the resulting boundaries. The legacy `rUECS` and `dilmarkers` functions remain in the source; `process_image` uses `split_clump`.
5. `Measurement` uses `minAreaRect` to estimate rod length and width, then derives shape descriptors per particle.
6. `Label overlay` assigns each detected particle its own colour (`_make_label_overlay`) so the per-particle segmentation can be inspected directly.

## Calibration (pixel size) sources

RodSizer resolves nm/pixel in this order; the first that succeeds wins:

1. **Manual override** (`manual_pixel_size`) — or the in-app "draw a line" / numeric calibration.
2. **Linked metadata file** — a matching `.dm3/.dm4/.emd` sharing the image's base name.
3. **Embedded metadata** — Gatan/DM, Velox/EMD, OME-TIFF, FEI/Thermo and ImageJ TIFF tags.
4. **Burned-in OCR** — for camera exports (JPG/PNG/TIF) with no metadata, RodSizer reads explicit pixel-size values in pm or estimates scale from the footer field of view divided by image width. It cross-checks the estimates when both are present and falls back to a drawn scale bar and its label. OCR requires Tesseract; see the [installation guide](docs/usage.md). If Tesseract is unavailable, OCR is skipped.
5. **Uncalibrated placeholder** — `1.0 nm/px`, flagged `is_placeholder`. Set a measured pixel size before interpreting the reported nanometer dimensions.

## Active parameters

| Parameter | Current Value / Behavior | What it does |
| :--- | :--- | :--- |
| `requested_bar_length_nm` | `200.0` nm default | Draws a synthetic blue scale bar if a real calibration is available and no override is given. |
| `manual_pixel_size` | Optional user override | Replaces metadata-based calibration when supplied. |
| `min_area_nm2` | `30` nm², with a floor of `500` pixels | Sets `min_size_px = max(500, int(30 / pixel_size_nm²))`; regions and measured contours below that area are rejected. |
| `region.solidity > 0.9` | Active | Treats highly convex objects as simple particles; lower-solidity objects are treated as clumps and split with the distance-transform watershed. |
| `binary_mask_tune` (separation slider) | `-6 .. +6`, default `0` | Drives both binarisation morphology (closing vs opening) and clump-splitting aggressiveness (watershed seed depth `h`). Higher = separate more; lower = merge more. |
| Overlap threshold | `0.15` | Used during non-maximum suppression to skip heavily overlapping duplicate detections. |
| Label overlay | Per-particle colours | Each accepted particle is drawn in its own colour so the segmentation can be verified visually. |

## Measurement and Descriptor Outputs

These values are computed for each accepted particle:

- `length_nm`
- `width_nm`
- `aspect_ratio`
- `volume_nm3`
- `orientation_deg`
- `area_px`
- `solidity`
- `convexity`
- `circularity`
- `eccentricity`

`aspect_ratio`, `solidity`, and `circularity` describe the measured particles. The interface exposes calibration and separation controls; these descriptors are not manual filter controls.

## Legacy parameters

The following items described in older documentation are not active as manual tuning parameters in the current `process_image` workflow:

- `target_dist_nm`
- `closing_radius_nm`
- manual threshold controls such as `Threshold`
- older separation/closing sliders or variables
- `aspect_ratio_px` as a direct hard filter
- `containment` as a direct hard filter

## Practical Effect of the Active Parameters

### 1. Minimum Area (`min_area_nm2`)
- Removes tiny specks, dirt, and segmentation noise.
- Increasing it makes detection more conservative.
- Decreasing it allows smaller objects through, but may increase false positives.

### 2. Solidity Split Threshold (`region.solidity > 0.9`)
- High-solidity objects are treated as single particles.
- Objects at or below the solidity threshold are sent to the distance-transform watershed in `split_clump`.
- Increasing this threshold makes the app split more borderline shapes as clumps.
- Decreasing this threshold makes the app keep more irregular shapes as single objects.

### 3. Manual Pixel Size (`manual_pixel_size`)
- Overrides image metadata calibration.
- Useful when metadata is missing or incorrect.
- A wrong value will directly affect all reported dimensions in nanometers.

### 4. Requested Scale Bar Length (`requested_bar_length_nm`)
- Controls the displayed synthetic scale bar length.
- Default is `200 nm`.
- This affects the visualization scale bar, not particle detection itself.

## Where To Adjust These Parameters

Edit the defaults in `process_image(...)` in [backend/processing.py](backend/processing.py).
