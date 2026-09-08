# Device Attribution API Handoff

This document describes the backend contract for integrating the device-attribution
feature into the main website. The main website should keep its existing frontend.
The standalone `web/` folder is only needed when running the MVP by itself.

## Frontend Tabs

The main website can expose two tabs under Device Attribution:

1. **Device Attribution**: upload one image. The backend checks camera metadata
   first and uses PRNU matching when metadata is unavailable or inconclusive.
2. **PRNU Fingerprint Lab**: upload several reference images from one known device
   and one unseen query image. The backend builds a temporary fingerprint and
   returns the match result.

## Endpoints

### `GET /api/status`

Returns whether the dataset/model is available.

Example response:

```json
{
  "ready": true,
  "images": 84,
  "devices": 3,
  "device_labels": ["oneplus_12r", "samsung_s23"]
}
```

### `POST /api/analyze`

Normal device-attribution endpoint.

Request: `multipart/form-data`

- `image`: one JPEG or PNG file

Response shape:

```json
{
  "device": {
    "label": "oneplus_12r",
    "method": "metadata",
    "confidence": 0.96
  },
  "metadata": {
    "make": "OnePlus",
    "model": "CPH2609",
    "software": null,
    "exif_present": true
  },
  "evidence": [
    "Camera make/model extracted from image metadata."
  ]
}
```

Expected method values are `metadata`, `prnu`, `weak_features`, or `inconclusive`.
The UI should display the returned `label` and method rather than inventing a device
name on the client.

### `POST /api/prnu-match`

PRNU Fingerprint Lab endpoint.

Request: `multipart/form-data`

- `references`: multiple JPEG or PNG files from one known device
- `query`: one unseen JPEG or PNG file

Response shape:

```json
{
  "device": {
    "label": "oneplus_12r",
    "method": "prnu",
    "confidence": 0.91
  },
  "match": {
    "correlation": 0.64,
    "references_used": 12,
    "status": "match"
  },
  "metadata": {
    "make": "OnePlus",
    "model": "CPH2609",
    "exif_present": true
  },
  "evidence": [
    "Query residual correlates with the temporary reference fingerprint."
  ]
}
```

If PRNU is weak, the endpoint should still return a device label using metadata or
the known reference label. It should report `method: "metadata_fallback"` instead of
showing a misleading `No PRNU Match` headline.

## Integration Notes

- Do not copy the standalone `web/` folder into the main website unless the team
  explicitly wants the standalone dashboard UI.
- Reuse the existing project’s FastAPI router and schema conventions.
- Keep the PRNU reference directory outside Git when it contains real forensic images.
- Store only approved reference fingerprints or model artifacts, not sensitive image
  evidence, in the repository.
- Treat all results as ranked evidence for the hackathon MVP, not absolute proof.

## Backend Source

The standalone implementation that demonstrates this contract is in:

- `prnu_attribution/api.py`
- `prnu_attribution/prnu.py`
- `prnu_attribution/metadata.py`
- `prnu_attribution/model.py`

The teammate can port the endpoint logic into the main project’s existing API layer
and connect the existing website tabs to these routes.
