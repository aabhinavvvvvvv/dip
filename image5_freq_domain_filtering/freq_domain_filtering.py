import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage import img_as_float

img    = img_as_float(camera())
F      = np.fft.fftshift(np.fft.fft2(img))
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

Y, X = np.ogrid[:rows, :cols]
dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)

radius = 30  # cutoff radius (same for both filters)

# ── Q1: High-Pass Filter ───────────────────────────────────────────────────
hpf_mask = (dist > radius).astype(np.float64)
F_hpf    = F * hpf_mask
img_hpf  = np.real(np.fft.ifft2(np.fft.ifftshift(F_hpf)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray');                          axes[0].set_title('Original Image')
axes[1].imshow(np.log1p(np.abs(F_hpf)), cmap='gray');     axes[1].set_title(f'HPF Spectrum\n(center radius={radius} zeroed)')
axes[2].imshow(img_hpf, cmap='gray');                      axes[2].set_title('High-Pass Filtered\n(edges enhanced)')
for ax in axes: ax.axis('off')
plt.suptitle('Q1: High-Pass Filter in Frequency Domain', fontsize=14)
plt.tight_layout()
plt.show()

# ── Q2: Low-Pass Filter ────────────────────────────────────────────────────
lpf_mask = (dist <= radius).astype(np.float64)
F_lpf    = F * lpf_mask
img_lpf  = np.real(np.fft.ifft2(np.fft.ifftshift(F_lpf)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray');                          axes[0].set_title('Original Image')
axes[1].imshow(np.log1p(np.abs(F_lpf)), cmap='gray');     axes[1].set_title(f'LPF Spectrum\n(only center radius={radius} kept)')
axes[2].imshow(img_lpf, cmap='gray');                      axes[2].set_title('Low-Pass Filtered\n(blurred / smoothed)')
for ax in axes: ax.axis('off')
plt.suptitle('Q2: Low-Pass Filter in Frequency Domain', fontsize=14)
plt.tight_layout()
plt.show()
