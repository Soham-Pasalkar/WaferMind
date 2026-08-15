document.addEventListener('DOMContentLoaded', () => {
  const restoredImage = document.getElementById('restored-image');

  const metricPsnr = document.getElementById('metric-psnr');
  const metricSsim = document.getElementById('metric-ssim');
  const metricLatency = document.getElementById('metric-latency');
  const metricResolution = document.getElementById('metric-resolution');

  const detailInputRes = document.getElementById('detail-input-res');
  const detailOutputRes = document.getElementById('detail-output-res');

  const downloadBtn = document.getElementById('download-btn');

  const storedResult = sessionStorage.getItem('restorationResult');

  if (!storedResult) {
    window.location.href = 'restore.html';
    return;
  }

  let result;

  try {
    result = JSON.parse(storedResult);
  } catch (error) {
    window.location.href = 'restore.html';
    return;
  }

  if (!result.restored_image_base64) {
    window.location.href = 'restore.html';
    return;
  }

  restoredImage.src =
    `data:image/png;base64,${result.restored_image_base64}`;

  metricPsnr.textContent =
    result.psnr !== null && result.psnr !== undefined
      ? `${Number(result.psnr).toFixed(2)} dB`
      : 'Unavailable';

  metricSsim.textContent =
    result.ssim !== null && result.ssim !== undefined
      ? Number(result.ssim).toFixed(4)
      : 'Unavailable';

  metricLatency.textContent =
    result.inference_time_ms !== null &&
    result.inference_time_ms !== undefined
      ? `${Number(result.inference_time_ms).toFixed(1)} ms`
      : 'Unavailable';

  metricResolution.textContent =
    result.output_resolution || 'Unavailable';

  detailInputRes.textContent =
    result.input_resolution || 'Unavailable';

  detailOutputRes.textContent =
    result.output_resolution || 'Unavailable';

  downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');

    link.href =
      `data:image/png;base64,${result.restored_image_base64}`;

    link.download = 'wafermind-restored.png';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
});