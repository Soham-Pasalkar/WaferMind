import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import NAFNetSR
from inference import calculate_metrics, encode_png, get_device, load_checkpoint, read_image, restore

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "checkpoints" / "best.pt"

app = FastAPI(title="WaferMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_device()
model = NAFNetSR(width=32)
load_checkpoint(model, CHECKPOINT, device)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "NAFNet",
        "device": str(device),
        "checkpoint": CHECKPOINT.name
    }


@app.post("/api/restore")
async def restore_image(
    degraded: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None)
):
    try:
        degraded_data = await degraded.read()
        degraded_array, input_size = read_image(degraded_data, degraded.filename or "")

        output_array, inference_time_ms = restore(model, degraded_array, device)
        output_png = encode_png(output_array)

        psnr = None
        ssim = None

        if ground_truth is not None:
            gt_data = await ground_truth.read()
            gt_array, _ = read_image(gt_data, ground_truth.filename or "")
            psnr, ssim = calculate_metrics(output_array, gt_array)

        encoded = base64.b64encode(output_png).decode("ascii")

        return {
            "restored_image_base64": encoded,
            "input_resolution": f"{input_size[0]} × {input_size[1]}",
            "output_resolution": f"{output_array.shape[1]} × {output_array.shape[0]}",
            "inference_time_ms": round(inference_time_ms, 3),
            "psnr": round(psnr, 4) if psnr is not None else None,
            "ssim": round(ssim, 6) if ssim is not None else None
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
