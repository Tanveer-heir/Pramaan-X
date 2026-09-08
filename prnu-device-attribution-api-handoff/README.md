# Device Attribution MVP

Local hackathon demo for identifying the likely source camera device from one image.
The device channel uses camera metadata when it is present and falls back to a trained
PRNU-style sensor-noise match when metadata is missing or stripped.

## Install

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset Layout

Use the device name as the folder name. Keep multiple reference images from each
device in `original_reference`, and keep separate test images in `original_test`:

```text
dataset/
  oneplus_12r/
    original_reference/
    original_test/
  samsung_s23/
    original_reference/
    original_test/
```

For useful PRNU matching, collect about 20-50 low-detail reference photos per device
and several different test photos. Do not send them through WhatsApp or another app
before placing them in the dataset.

## Train On Your Devices

```powershell
python -m prnu_attribution.train --dataset dataset --model-dir models
```

## Run The Device Dashboard

```powershell
python -m prnu_attribution.api --host 127.0.0.1 --port 8002 --dataset dataset --model-dir models
```

Open `http://127.0.0.1:8002` in a browser, choose one image, and click **Analyze Device**.

The result reports:

- the predicted device label from the trained dataset;
- the method used: metadata, PRNU, weak file features, or inconclusive;
- confidence and supporting evidence;
- extracted EXIF/JPEG information.

## PRNU Fingerprint Lab

Open the `PRNU Fingerprint Lab` tab in the dashboard when a forensics team has a
reference set from one known device and wants to test an unseen image.

1. Upload multiple reference images from the known device.
2. Upload one unseen query image.
3. Click **Build Fingerprint & Match**.

The lab averages the reference residuals into a temporary PRNU fingerprint and reports
the query correlation, confidence, usable reference count, and identified device. It
uses PRNU when the signal is strong, then falls back to query metadata, reference
metadata, or the selected folder label. The temporary fingerprint is not added to the
global model unless the team separately trains the dataset.

## Quick Smoke Test

This generates artificial images only to prove that the application is wired correctly:

```powershell
python scripts/make_sample_dataset.py --out sample_dataset
python -m prnu_attribution.train --dataset sample_dataset --model-dir models
python -m prnu_attribution.api --host 127.0.0.1 --port 8002 --dataset sample_dataset --model-dir models --allow-demo-model
```

The synthetic device names are not real phone identifications. Train on real phone
images before making a forensic or hackathon claim.

## Command-Line Prediction

```powershell
python -m prnu_attribution.predict --image path\to\query.jpg --model-dir models
```

## Forensic Framing

Metadata can directly reveal make/model when it survives file handling. Social apps,
messengers, screenshots, and editing tools may strip or rewrite it. PRNU is useful for
matching against a known reference device, but it cannot identify an arbitrary new
phone unless that phone has a reference fingerprint in the trained dataset.

The output is ranked evidence for a hackathon MVP, not an absolute forensic conclusion.
