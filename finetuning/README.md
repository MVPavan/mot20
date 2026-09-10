# Detector Fine-Tuning

This directory owns the RF-DETR MOT20 detector-training workflow: dataset
conversion, ignored-region loss integration, tests, launchers, plans, and
preflight evidence.

Run its lightweight checks from the repository root:

```bash
make -C finetuning test
make -C finetuning compile
```

The conversion launcher requires explicit dataset paths and a new output
directory, and it never extracts archives or overwrites manifest artifacts:

```bash
PYTHONPATH=finetuning/src .venv/bin/python \
  finetuning/scripts/convert_bytetrack_mot20_crowdhuman.py --help
```

The RF-DETR launcher reads a versioned configuration, re-audits the supplied
dataset, and writes run provenance only for an approved configuration. The
checked-in config is intentionally blocked until real conversion and capacity
evidence update it:

```bash
PYTHONPATH=finetuning/src .venv/bin/python \
  finetuning/scripts/train_rfdetr_2xl.py --help
```

The separately auditable Byte65 source dataset is materialized explicitly from
post-modification YOLO labels:

```bash
PYTHONPATH=finetuning/src .venv/bin/python \
  finetuning/scripts/materialize_byte65_yolo_dataset.py --help
```

Extract CrowdHuman only through the guarded command, with a new destination:

```bash
PYTHONPATH=finetuning/src .venv/bin/python \
  finetuning/scripts/extract_crowdhuman.py --help
```

Before conversion or real training, read the authoritative plan in
`finetuning/docs/plans/2026-09-03-rfdetr-2xl-mot20-training.md`.