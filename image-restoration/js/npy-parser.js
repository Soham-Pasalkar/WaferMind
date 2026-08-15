/**
 * npy-parser.js
 * Minimal in-browser parser for NumPy .npy files, plus a helper that
 * renders the parsed array onto a <canvas> so it can be previewed and
 * treated like any other uploaded image in the WaferMind pipeline.
 *
 * Supports the common dtypes produced by wafer-map / sensor-array exports:
 * uint8, int8, uint16, int16, uint32, int32, float32, float64, bool.
 * Supports 2D (H, W) and 3D (H, W, C) arrays, and will squeeze a leading
 * singleton batch dimension e.g. (1, H, W) or (1, H, W, C).
 */
(function (global) {
  const DTYPE_MAP = {
    'u1': { bytes: 1, read: (dv, o) => dv.getUint8(o) },
    'i1': { bytes: 1, read: (dv, o) => dv.getInt8(o) },
    'b1': { bytes: 1, read: (dv, o) => dv.getUint8(o) },
    'u2': { bytes: 2, read: (dv, o, le) => dv.getUint16(o, le) },
    'i2': { bytes: 2, read: (dv, o, le) => dv.getInt16(o, le) },
    'u4': { bytes: 4, read: (dv, o, le) => dv.getUint32(o, le) },
    'i4': { bytes: 4, read: (dv, o, le) => dv.getInt32(o, le) },
    'f4': { bytes: 4, read: (dv, o, le) => dv.getFloat32(o, le) },
    'f8': { bytes: 8, read: (dv, o, le) => dv.getFloat64(o, le) },
  };

  function parseHeader(bytes) {
    const magic = String.fromCharCode(...bytes.slice(0, 6));
    if (magic !== '\x93NUMPY') {
      throw new Error('Not a valid .npy file (bad magic number).');
    }
    const major = bytes[6];
    let headerLen, headerStart;
    if (major === 1) {
      headerLen = bytes[8] | (bytes[9] << 8);
      headerStart = 10;
    } else {
      headerLen = (bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24)) >>> 0;
      headerStart = 12;
    }
    const headerStr = String.fromCharCode(...bytes.slice(headerStart, headerStart + headerLen));
    const dataStart = headerStart + headerLen;

    const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
    const orderMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
    const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);

    if (!descrMatch || !shapeMatch) {
      throw new Error('Could not parse .npy header.');
    }

    const descr = descrMatch[1];
    const littleEndian = descr[0] !== '>';
    const typeCode = descr.replace(/^[<>|=]/, '');
    const fortranOrder = orderMatch ? orderMatch[1] === 'True' : false;
    const shape = shapeMatch[1]
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => parseInt(s, 10));

    return { typeCode, littleEndian, fortranOrder, shape, dataStart };
  }

  /**
   * Parses an ArrayBuffer containing .npy data.
   * Returns { data: Float64Array, shape: number[], dtype: string }
   */
  function parseNpy(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    const { typeCode, littleEndian, fortranOrder, shape, dataStart } = parseHeader(bytes);

    const typeInfo = DTYPE_MAP[typeCode];
    if (!typeInfo) {
      throw new Error(`Unsupported .npy dtype: "${typeCode}". Try exporting as uint8/float32.`);
    }
    if (fortranOrder) {
      throw new Error('Fortran-ordered .npy arrays are not supported. Please save with order="C".');
    }

    const count = shape.reduce((a, b) => a * b, 1);
    const dv = new DataView(arrayBuffer, dataStart);
    const out = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = typeInfo.read(dv, i * typeInfo.bytes, littleEndian);
    }

    return { data: out, shape, dtype: typeCode };
  }

  /**
   * Normalizes a flat array's values into 0-255 using min-max scaling.
   * Booleans / uint8 in [0,255] are passed through unscaled when already
   * in range; everything else (float arrays, sensor counts, etc.) is
   * rescaled so the full dynamic range is visible.
   */
  function normalizeTo255(data, dtype) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (dtype === 'u1' && min >= 0 && max <= 255) {
      return data; // already displayable
    }
    const range = max - min || 1;
    const out = new Float64Array(data.length);
    for (let i = 0; i < data.length; i++) {
      out[i] = ((data[i] - min) / range) * 255;
    }
    return out;
  }

  // Simple perceptual "inferno-ish" colormap for single-channel wafer maps,
  // so defect intensity is easier to read than flat grayscale.
  function colormap(t) {
    // t in [0,1]
    const stops = [
      [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
      [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
    ];
    const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
    const idx = Math.floor(scaled);
    const frac = scaled - idx;
    const a = stops[idx];
    const b = stops[Math.min(idx + 1, stops.length - 1)];
    return [
      Math.round(a[0] + (b[0] - a[0]) * frac),
      Math.round(a[1] + (b[1] - a[1]) * frac),
      Math.round(a[2] + (b[2] - a[2]) * frac),
    ];
  }

  /**
   * Renders a parsed .npy array onto a canvas and returns the canvas.
   * Handles (H,W), (H,W,1), (H,W,3), (H,W,4), and squeezes a leading
   * singleton dim like (1,H,W) or (1,H,W,C).
   */
  function npyToCanvas(parsed, opts = {}) {
    let { data, shape, dtype } = parsed;
    let dims = shape.slice();

    if (dims.length === 4 && dims[0] === 1) dims = dims.slice(1);
    if (dims.length === 3 && dims[0] === 1) dims = dims.slice(1);

    let height, width, channels;
    if (dims.length === 2) {
      [height, width] = dims;
      channels = 1;
    } else if (dims.length === 3) {
      [height, width, channels] = dims;
    } else {
      throw new Error(`Unsupported array shape [${shape.join(', ')}]. Expected 2D or 3D (H,W[,C]).`);
    }
    if (!height || !width) {
      throw new Error(`Array has an empty dimension: [${shape.join(', ')}].`);
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);

    const useColormap = channels === 1 && opts.colormap !== false;
    const norm = normalizeTo255(data, dtype);

    for (let p = 0; p < width * height; p++) {
      let r, g, b, a = 255;
      if (channels === 1) {
        const v = norm[p];
        if (useColormap) {
          [r, g, b] = colormap(v / 255);
        } else {
          r = g = b = v;
        }
      } else if (channels === 3) {
        r = norm[p * 3]; g = norm[p * 3 + 1]; b = norm[p * 3 + 2];
      } else if (channels === 4) {
        r = norm[p * 4]; g = norm[p * 4 + 1]; b = norm[p * 4 + 2]; a = norm[p * 4 + 3];
      } else {
        // Fallback: use first channel as grayscale
        const v = norm[p * channels];
        r = g = b = v;
      }
      const o = p * 4;
      imgData.data[o] = r;
      imgData.data[o + 1] = g;
      imgData.data[o + 2] = b;
      imgData.data[o + 3] = a;
    }

    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  global.NpyParser = { parseNpy, npyToCanvas };
})(window);
