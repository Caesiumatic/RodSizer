# RodSizer

RodSizer turns electron-microscopy images into reviewed nanoparticle
measurements and size distributions. It brings detection, calibration,
particle selection, and reporting into one local application.

## Measurement workflow

Upload TEM, STEM, or SEM images, calibrate the scale, and inspect the detected
particles. Selections can be adjusted before generating measurements and
combining results across an image folder.

Calibration comes from embedded image metadata, scale-bar OCR, or a manually
drawn reference. The review interface lets users inspect particle boundaries
and exclude aggregates or ambiguous detections.

## Method

The image-analysis pipeline uses K-means thresholding to identify foreground
particles and distance-transform watershed to separate touching regions.
Rotated bounding boxes provide length, width, aspect ratio, and orientation;
additional measurements describe particle area and shape.

Volume is estimated with a hemispherically capped cylinder model. Segmentation
builds on AutoDetect-mNP (JACS Au, 2021, 1, 316–327), with watershed splitting and
OCR calibration integrated into the measurement workflow.

## Outputs

RodSizer produces particle overlays, per-particle measurements, size
distributions, and folder-level statistics. CSV, Excel, and PDF exports carry
the reviewed selections and calibration into subsequent analysis.

## Documentation

- [Installation, local use, and Docker](docs/usage.md)
- [Detection and measurement parameters](PARAMETER_GUIDE.md)

## Contributors

Shi Chen and Arda Turk —
[Murphy Group, UIUC](https://murphy-group.chemistry.illinois.edu/).
