# Proposed Prediction API Design

This document describes a **proposed API only**. The project does not currently
implement or serve these endpoints.

The model is trained on EuroSAT RGB images. The existing dataset loader converts
images to RGB tensors and normalizes each channel using mean and standard
deviation `0.5`. EuroSAT RGB images are 64×64 pixels and use these class names:
`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`,
`Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.

## `GET /health`

Checks whether the prediction service is running and ready to load the model.

- **Method:** `GET`
- **Endpoint:** `/health`
- **Success status:** `200 OK`

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

If the service is running but the model is unavailable, it should return
`503 Service Unavailable`, for example:

```json
{
  "error": "model_unavailable",
  "message": "The prediction model is not loaded."
}
```

## `POST /predict`

Accepts one EuroSAT RGB image and returns its predicted land-cover class.

- **Method:** `POST`
- **Endpoint:** `/predict`
- **Content type:** `application/json`
- **Input:** A base64-encoded JPEG or PNG RGB image. The image should be 64×64
  pixels, matching the EuroSAT training images. The service should decode it,
  convert it to RGB, and apply the same tensor conversion and normalization
  used by the training data loader.

Example request (replace the placeholder with the base64 data for a real
64×64 EuroSAT RGB image):

```json
{
  "image_base64": "<base64-encoded 64x64 RGB JPEG or PNG>"
}
```

Example success response:

```json
{
  "predicted_class": "Forest",
  "confidence": 0.91,
  "probabilities": {
    "AnnualCrop": 0.01,
    "Forest": 0.91,
    "HerbaceousVegetation": 0.02,
    "Highway": 0.01,
    "Industrial": 0.01,
    "Pasture": 0.01,
    "PermanentCrop": 0.01,
    "Residential": 0.01,
    "River": 0.01,
    "SeaLake": 0.0
  }
}
```

The response values above illustrate the format and are not a measured model
prediction. `predicted_class` must be one of the model's ten class names;
`confidence` is the predicted class's probability. Return probabilities from a
softmax over the model's class scores; they should sum to approximately 1.

Suggested error responses:

| Status | When to use it | Example response |
| --- | --- | --- |
| `400 Bad Request` | The JSON is missing `image_base64` or contains invalid base64. | `{"error":"invalid_request","message":"Provide a base64-encoded image in 'image_base64'."}` |
| `415 Unsupported Media Type` | The request is not JSON or the decoded file is not a supported image. | `{"error":"unsupported_image","message":"Provide a JPEG or PNG image."}` |
| `422 Unprocessable Entity` | The decoded image is not 64×64 pixels. | `{"error":"invalid_image_size","message":"Expected a 64x64 RGB image."}` |
| `503 Service Unavailable` | The model is not loaded or ready. | `{"error":"model_unavailable","message":"The prediction model is not loaded."}` |
