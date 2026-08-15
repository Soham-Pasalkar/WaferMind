document.addEventListener('DOMContentLoaded', () => {
  const degradedDropzone = document.getElementById('degraded-dropzone');
  const degradedFileInput = document.getElementById('degraded-file-input');
  const degradedPrompt = document.getElementById('degraded-prompt');
  const degradedPreviewContainer = document.getElementById('degraded-preview-container');
  const degradedPreviewImg = document.getElementById('degraded-preview-img');
  const degradedFileMeta = document.getElementById('degraded-file-meta');
  const degradedError = document.getElementById('degraded-error');
  const changeDegradedBtn = document.getElementById('change-degraded');
  const removeDegradedBtn = document.getElementById('remove-degraded');

  const gtDropzone = document.getElementById('groundtruth-dropzone');
  const gtFileInput = document.getElementById('groundtruth-file-input');
  const gtPrompt = document.getElementById('groundtruth-prompt');
  const gtPreviewContainer = document.getElementById('groundtruth-preview-container');
  const gtPreviewImg = document.getElementById('groundtruth-preview-img');
  const gtFileMeta = document.getElementById('groundtruth-file-meta');
  const gtError = document.getElementById('groundtruth-error');
  const removeGtBtn = document.getElementById('remove-groundtruth');

  const restoreBtn = document.getElementById('restore-btn');
  const processingOverlay = document.getElementById('processing-overlay');
  const progressBar = document.getElementById('progress-bar');
  const processingSteps = document.querySelectorAll('.processing-steps .step');

  const API_URL = 'http://127.0.0.1:8000/api/restore';

  let degradedFile = null;
  let groundTruthFile = null;

  const ACCEPTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.npy'];

  degradedDropzone.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    degradedFileInput.click();
  });

  degradedDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    degradedDropzone.classList.add('dragover');
  });

  degradedDropzone.addEventListener('dragleave', () => {
    degradedDropzone.classList.remove('dragover');
  });

  degradedDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    degradedDropzone.classList.remove('dragover');

    if (e.dataTransfer.files.length) {
      handleDegradedFile(e.dataTransfer.files[0]);
    }
  });

  degradedFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleDegradedFile(e.target.files[0]);
    }
  });

  changeDegradedBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    degradedFileInput.click();
  });

  removeDegradedBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetDegraded();
  });

  gtDropzone.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    gtFileInput.click();
  });

  gtDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    gtDropzone.classList.add('dragover');
  });

  gtDropzone.addEventListener('dragleave', () => {
    gtDropzone.classList.remove('dragover');
  });

  gtDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    gtDropzone.classList.remove('dragover');

    if (e.dataTransfer.files.length) {
      handleGtFile(e.dataTransfer.files[0]);
    }
  });

  gtFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleGtFile(e.target.files[0]);
    }
  });

  removeGtBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetGt();
  });

  restoreBtn.addEventListener('click', restoreImage);

  function handleDegradedFile(file) {
    setError(degradedError, '');

    if (!isSupportedFile(file)) {
      setError(degradedError, 'Unsupported file type. Please upload PNG, JPG, JPEG, or NPY.');
      return;
    }

    degradedFile = file;

    const reader = file.name.toLowerCase().endsWith('.npy')
      ? readNpyFile(file)
      : readImageFile(file);

    reader
      .then(({ dataUrl, resolution }) => {
        showDegradedPreview(dataUrl, file.name, resolution);
      })
      .catch((err) => {
        degradedFile = null;
        setError(degradedError, err.message || 'Could not read that file.');
      });
  }

  function handleGtFile(file) {
    setError(gtError, '');

    if (!isSupportedFile(file)) {
      setError(gtError, 'Unsupported file type. Please upload PNG, JPG, JPEG, or NPY.');
      return;
    }

    groundTruthFile = file;

    const reader = file.name.toLowerCase().endsWith('.npy')
      ? readNpyFile(file)
      : readImageFile(file);

    reader
      .then(({ dataUrl }) => {
        showGtPreview(dataUrl, file.name);
      })
      .catch((err) => {
        groundTruthFile = null;
        setError(gtError, err.message || 'Could not read that file.');
      });
  }

  async function restoreImage() {
    if (!degradedFile) {
      setError(degradedError, 'Please select an image first.');
      return;
    }

    restoreBtn.disabled = true;
    processingOverlay.style.display = 'flex';
    resetProgress();

    try {
      updateProgress(20, 0);

      const formData = new FormData();
      formData.append('degraded', degradedFile);

      if (groundTruthFile) {
        formData.append('ground_truth', groundTruthFile);
      }

      updateProgress(40, 1);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });

      updateProgress(70, 2);

      if (!response.ok) {
        let message = 'Restoration failed.';

        try {
          const errorData = await response.json();
          message = errorData.detail || message;
        } catch (_) {}

        throw new Error(message);
      }

      const result = await response.json();

      if (!result.restored_image_base64) {
        throw new Error('The server did not return a restored image.');
      }

      updateProgress(90, 3);

      sessionStorage.setItem(
        'restorationResult',
        JSON.stringify(result)
      );

      updateProgress(100, 4);

      setTimeout(() => {
        window.location.href = 'results.html';
      }, 350);

    } catch (error) {
      processingOverlay.style.display = 'none';
      restoreBtn.disabled = false;
      setError(degradedError, error.message || 'Could not connect to the restoration server.');
    }
  }

  function resetProgress() {
    progressBar.style.width = '0%';

    processingSteps.forEach((step) => {
      step.classList.remove('active');
    });
  }

  function updateProgress(progress, stepIndex) {
    progressBar.style.width = `${progress}%`;

    processingSteps.forEach((step, index) => {
      step.classList.toggle('active', index <= stepIndex);
    });
  }

  function showDegradedPreview(dataUrl, name, resolution) {
    degradedPreviewImg.src = dataUrl;

    degradedFileMeta.textContent =
      `${name}${resolution ? ` • ${resolution}` : ''}`;

    degradedPrompt.style.display = 'none';
    degradedPreviewContainer.style.display = 'flex';

    restoreBtn.disabled = false;
  }

  function showGtPreview(dataUrl, name) {
    gtPreviewImg.src = dataUrl;
    gtFileMeta.textContent = `${name} (Ground Truth ✓)`;

    gtPrompt.style.display = 'none';
    gtPreviewContainer.style.display = 'flex';
  }

  function resetDegraded() {
    degradedFile = null;

    degradedPreviewImg.src = '';
    degradedFileInput.value = '';

    degradedPreviewContainer.style.display = 'none';
    degradedPrompt.style.display = 'block';

    setError(degradedError, '');

    restoreBtn.disabled = true;
  }

  function resetGt() {
    groundTruthFile = null;

    gtPreviewImg.src = '';
    gtFileInput.value = '';

    gtPreviewContainer.style.display = 'none';
    gtPrompt.style.display = 'block';

    setError(gtError, '');
  }

  function isSupportedFile(file) {
    const name = file.name.toLowerCase();

    return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
  }

  function readImageFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onerror = () => {
        reject(new Error('Could not read that image.'));
      };

      reader.onload = (event) => {
        const img = new Image();

        img.onerror = () => {
          reject(new Error('That image could not be decoded.'));
        };

        img.onload = () => {
          const width = img.width;
          const height = img.height;

          resolve({
            dataUrl: event.target.result,
            resolution: `${width} × ${height}`
          });
        };

        img.src = event.target.result;
      };

      reader.readAsDataURL(file);
    });
  }

  function readNpyFile(file) {
    return new Promise((resolve, reject) => {
      if (typeof NpyParser === 'undefined') {
        reject(new Error('NPY parser is not available.'));
        return;
      }

      const reader = new FileReader();

      reader.onerror = () => {
        reject(new Error('Could not read that NPY file.'));
      };

      reader.onload = (event) => {
        try {
          const parsed = NpyParser.parseNpy(event.target.result);
          const canvas = NpyParser.npyToCanvas(parsed);

          resolve({
            dataUrl: canvas.toDataURL('image/png'),
            resolution: `${canvas.width} × ${canvas.height}`
          });
        } catch (error) {
          reject(error);
        }
      };

      reader.readAsArrayBuffer(file);
    });
  }

  function setError(element, message) {
    if (!element) return;

    element.textContent = message;
    element.style.display = message ? 'block' : 'none';
  }
});