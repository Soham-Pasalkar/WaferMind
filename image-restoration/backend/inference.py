import io
import time
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return checkpoint


def read_image(data, filename):
    if filename.lower().endswith(".npy"):
        array = np.load(io.BytesIO(data))
        array = np.asarray(array)

        if array.ndim == 3:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[-1] == 1:
                array = array[..., 0]
            else:
                raise ValueError("Only grayscale .npy arrays are supported.")

        if array.ndim != 2:
            raise ValueError("Expected a 2D grayscale .npy array.")

        return array.astype(np.float32), (array.shape[1], array.shape[0])

    image = Image.open(io.BytesIO(data)).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array, image.size


def array_to_tensor(array, device):
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def encode_png(array):
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def calculate_metrics(pred, target):
    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    target = np.clip(target, 0.0, 1.0).astype(np.float32)

    if pred.shape != target.shape:
        target_image = Image.fromarray((target * 255).round().astype(np.uint8))
        target_image = target_image.resize(
            (pred.shape[1], pred.shape[0]),
            Image.Resampling.BICUBIC
        )
        target = np.asarray(target_image, dtype=np.float32) / 255.0

    psnr = peak_signal_noise_ratio(target, pred, data_range=1.0)
    ssim = structural_similarity(target, pred, data_range=1.0)
    return float(psnr), float(ssim)


@torch.inference_mode()
def restore(model, array, device):
    tensor = array_to_tensor(array, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    output = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output = output.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    return output, elapsed_ms
