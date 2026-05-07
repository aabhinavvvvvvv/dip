import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage import img_as_float

img     = img_as_float(camera())   # float [0, 1]
img_u8  = camera()                 # uint8 [0, 255]

# ── Q1: Image Negative, Log Transform, Power-Law Transform ────────────────
img_negative = 1.0 - img

c       = 1.0 / np.log(1 + np.max(img))
img_log = c * np.log(1 + img)

gammas     = [0.4, 1.0, 2.5]
imgs_gamma = [np.clip(img ** g, 0, 1) for g in gammas]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0, 0].imshow(img, cmap='gray');          axes[0, 0].set_title('Original')
axes[0, 1].imshow(img_negative, cmap='gray'); axes[0, 1].set_title('Image Negative')
axes[0, 2].imshow(img_log, cmap='gray');      axes[0, 2].set_title('Log Transform')
for i, (g, im) in enumerate(zip(gammas, imgs_gamma)):
    lbl = 'brightens' if g < 1 else ('no change' if g == 1 else 'darkens')
    axes[1, i].imshow(im, cmap='gray')
    axes[1, i].set_title(f'Power-Law γ={g}  [{lbl}]')
for ax in axes.ravel(): ax.axis('off')
plt.suptitle('Q1: Image Negative, Log Transform, Power-Law Transform', fontsize=14)
plt.tight_layout()
plt.show()

# ── Q2: Contrast Stretching and Thresholding ──────────────────────────────
r_min = np.percentile(img, 2)
r_max = np.percentile(img, 98)
img_stretched = np.clip((img - r_min) / (r_max - r_min), 0, 1)

T          = 0.5
img_binary = (img >= T).astype(np.float64)

imgs2   = [img, img_stretched, img_binary]
titles2 = ['Original',
           f'Contrast Stretched\n[{r_min:.2f}, {r_max:.2f}] → [0, 1]',
           f'Threshold (T={T})']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, image, title in zip(axes[0], imgs2, titles2):
    ax.imshow(image, cmap='gray'); ax.set_title(title); ax.axis('off')
for ax, image, title in zip(axes[1], imgs2, titles2):
    data = (image * 255).astype(np.uint8)
    ax.hist(data.ravel(), bins=256, color='steelblue', edgecolor='none')
    ax.set_title(f'{title.split(chr(10))[0]} — Histogram')
    ax.set_xlabel('Pixel Value'); ax.set_ylabel('Frequency')
plt.suptitle('Q2: Contrast Stretching and Thresholding', fontsize=14)
plt.tight_layout()
plt.show()

# ── Q3: Bit-Plane Slicing and Gray Level Slicing ──────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Q3a: Bit-Plane Slicing  (Bit 0 = LSB,  Bit 7 = MSB)', fontsize=14)
for bit in range(8):
    ax    = axes[bit // 4, bit % 4]
    plane = ((img_u8 >> bit) & 1) * 255
    label = '  [MSB]' if bit == 7 else ('  [LSB]' if bit == 0 else '')
    ax.imshow(plane, cmap='gray'); ax.set_title(f'Bit {bit}{label}'); ax.axis('off')
plt.tight_layout()
plt.show()

low, high = 100, 180
img_highlight    = np.where((img_u8 >= low) & (img_u8 <= high), 255, img_u8)
img_binary_slice = np.where((img_u8 >= low) & (img_u8 <= high), 255, 0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_u8, cmap='gray');           axes[0].set_title('Original')
axes[1].imshow(img_highlight, cmap='gray');    axes[1].set_title(f'Gray Level Slicing [{low}–{high}]\nWith Background')
axes[2].imshow(img_binary_slice, cmap='gray'); axes[2].set_title(f'Gray Level Slicing [{low}–{high}]\nWithout Background')
for ax in axes: ax.axis('off')
plt.suptitle('Q3b: Gray Level Slicing', fontsize=14)
plt.tight_layout()
plt.show()
