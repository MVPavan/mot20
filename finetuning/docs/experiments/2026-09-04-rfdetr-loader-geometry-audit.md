# RF-DETR Aspect-Preserving Loader Geometry Audit

## Scope

This CPU-only audit exercised the installed RF-DETR 1.9.4 data pipeline against
every image in the immutable clean MOT20/CrowdHuman dataset. It constructed
only `RFDETR2XLargeConfig`, `TrainConfig`, the data module, COCO datasets,
transforms, and collator. It did not construct an RF-DETR model, load a
checkpoint, execute a forward pass, or use a GPU.

## Configuration

The audited clean configuration is
`finetuning/configs/rfdetr_2xl_clean_ddp_batch8_lr5e5_characterization.toml`:

- `resolution = 1120`
- `multi_scale = true`
- `do_random_resize_via_padding = true`
- `square_resize_div_64 = false`
- `scale_jitter = false`
- 2XL collator block size: 40px

The loader's configured training short-side targets were 920 through 1320px in
40px increments. Validation used the configured 1120px short side.

## Results

The completed receipt is
`finetuning/artifacts/rfdetr-loader-geometry-aspect-preserving-1120-2026-09-04-r2.json`.

| Split | Images | Sequential batches | Max ordinary / ignored boxes | Max transformed long side | Max padded shape (W x H) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 23,838 | 2,980 | 377 / 169 | 1335 | 1360 x 1360 |
| valid | 4,463 | 558 | 220 / 28 | 1333 | 1360 x 1000 |

For every item, the transformed dimensions exactly matched the installed
`RandomResize._get_size` calculation for one configured target; ordinary and
ignored target counts matched the source COCO annotations. For every sequential
batch, the collator produced the minimal batch-maximum H/W rounded up to a
multiple of 40, without cropping a transformed image.

The train pass observed each of the eleven configured target scales in at least
one unambiguous image. It also found 21,634 images whose capped output can be
produced by more than one requested scale. Those cap-ambiguous outputs cannot
identify a unique randomly selected scale, so the receipt does not present
their scale matches as selection frequencies.

## Cap Finding

The intended cap is 1333px, but the installed training `RandomResize` can
produce a 1334px or 1335px long edge. Its calculation rounds the cap-derived
short side to an integer and then rounds the corresponding long side, allowing
up to a two-pixel overshoot. Validation's fixed 1120px transform did not exceed
1333px. This is a verified installed-library behavior, not square warping or a
crop.

Consequently, this audit verifies aspect preservation, direct resizing, target
retention, and 40px batch padding, but it does **not** verify a strict 1333px
training long-edge bound. A fresh aspect-preserving capacity probe must use the
observed 1360 x 1360 padded envelope, or the installed resize implementation
must be changed and the complete audit rerun.