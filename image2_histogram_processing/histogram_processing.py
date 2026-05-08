import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.data import camera, moon

img = camera()
ref = moon()

# ── Q1: Histogram Equalization and Histogram Specification ─────────────────
img_eq      = exposure.equalize_hist(img)
img_matched = exposure.match_histograms(img, ref)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
titles = ['Original', 'Histogram Equalized', 'Histogram Specified\n(matched to Moon)']
imgs   = [img, img_eq, img_matched]

for ax, image, title in zip(axes[0], imgs, titles):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for ax, image, title in zip(axes[1], imgs, titles):
    data = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    ax.hist(data.ravel(), bins=256, color='steelblue', edgecolor='none')
    ax.set_title(f'{title.split(chr(10))[0]} — Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')

plt.suptitle('Q1: Histogram Equalization and Specification', fontsize=14)
plt.tight_layout()
plt.show()

# ── Q2: CLAHE ──────────────────────────────────────────────────────────────
img_he    = exposure.equalize_hist(img)
img_clahe = exposure.equalize_adapthist(img, clip_limit=0.03)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
imgs2   = [img, img_he, img_clahe]
titles2 = ['Original', 'Histogram Equalization', 'CLAHE (clip_limit=0.03)']
colors  = ['blue', 'orange', 'green']

for ax, image, title in zip(axes[0], imgs2, titles2):
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

for ax, image, title, color in zip(axes[1], imgs2, titles2, colors):
    data = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image
    ax.hist(data.ravel(), bins=256, color=color, alpha=0.8, edgecolor='none')
    ax.set_title(f'{title} — Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')

plt.suptitle('Q2: CLAHE vs Standard Histogram Equalization', fontsize=14)
plt.tight_layout()
plt.show()
