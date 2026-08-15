# WaferMind

NAFNet-based image restoration application for grayscale degraded images.

## Structure

```text
wafermind/
├── backend/
│   ├── app.py
│   ├── inference.py
│   ├── model.py
│   ├── requirements.txt
│   └── checkpoints/
│       └── best.pt
├── evaluation/
│   └── evaluate.py
└── frontend_integration/
    ├── restore.js
    └── results.js
```

Keep the existing HTML, CSS and assets from the frontend in place. Replace only `restore.js` and `results.js` with the files in `frontend_integration`.

## Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the trained checkpoint:

```text
backend/checkpoints/best.pt
```

Start the API:

```bash
uvicorn app:app --reload --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

## Frontend

From the frontend directory:

```bash
python3 -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

## API

`POST /api/restore`

Form fields:

- `degraded`: required
- `ground_truth`: optional

Supported input formats:

- PNG
- JPEG
- NPY

The response contains:

- restored PNG as base64
- input resolution
- output resolution
- inference time
- PSNR when ground truth is supplied
- SSIM when ground truth is supplied

## Test-set evaluation

```bash
python3 evaluation/evaluate.py \
  --test_dir /path/to/test \
  --output_dir /path/to/restored_outputs \
  --checkpoint backend/checkpoints/best.pt
```

The evaluation script accepts the `.npy` test images and writes PNG outputs.

## Model

The backend uses the same 5.48M-parameter NAFNetSR configuration used during training:

- width: 32
- encoder blocks: 2, 2, 4
- decoder blocks: 2, 2, 2
- grayscale input
- 2x output resolution
