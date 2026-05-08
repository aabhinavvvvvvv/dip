import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera, brick, grass
from skimage import img_as_float

# ── Q1: 2D DFT — visualize magnitude, phase, IDFT reconstruction ──────────
img = img_as_float(camera())

F          = np.fft.fft2(img)
F_shifted  = np.fft.fftshift(F)
mag_log    = np.log1p(np.abs(F_shifted))
phase      = np.angle(F_shifted)
img_recon  = np.real(np.fft.ifft2(np.fft.ifftshift(F_shifted)))

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(img, cmap='gray');       axes[0].set_title('a) Original Image')
axes[1].imshow(mag_log, cmap='gray');   axes[1].set_title('b) DFT Magnitude (log-scaled)')
axes[2].imshow(phase, cmap='hsv');      axes[2].set_title('c) DFT Phase')
axes[3].imshow(img_recon, cmap='gray'); axes[3].set_title('d) Reconstructed (IDFT)')
for ax in axes: ax.axis('off')
plt.suptitle('Q1: 2D DFT Visualization', fontsize=14)
plt.tight_layout()
plt.show()
print(f"Q1 Max reconstruction error: {np.max(np.abs(img - img_recon)):.2e}")

# ── Q2: DFT Basis Images (real & imaginary) for 8×8 and 16×16 ─────────────
def dft_basis(N):
    x = np.arange(N)
    xx, yy = np.meshgrid(x, x)
    real_b = np.zeros((N, N, N, N))
    imag_b = np.zeros((N, N, N, N))
    for u in range(N):
        for v in range(N):
            e = np.exp(2j * np.pi * (u * xx + v * yy) / N)
            real_b[u, v] = np.real(e)
            imag_b[u, v] = np.imag(e)
    return real_b, imag_b

for N in [8, 16]:
    rb, ib = dft_basis(N)
    fig, axes = plt.subplots(2, N, figsize=(N * 1.2, 3.5))
    fig.suptitle(f'Q2: DFT Basis Images ({N}×{N})  |  Top: Real   Bottom: Imaginary', fontsize=11)
    for v in range(N):
        axes[0, v].imshow(rb[0, v], cmap='gray', vmin=-1, vmax=1)
        axes[0, v].set_title(f'v={v}', fontsize=6)
        axes[0, v].axis('off')
        axes[1, v].imshow(ib[0, v], cmap='gray', vmin=-1, vmax=1)
        axes[1, v].axis('off')
    plt.tight_layout()
    plt.show()

# ── Q3: Swap magnitude and phase between two texture images ────────────────
img1 = img_as_float(brick())[:256, :256]
img2 = img_as_float(grass())[:256, :256]

F1 = np.fft.fftshift(np.fft.fft2(img1))
F2 = np.fft.fftshift(np.fft.fft2(img2))

mag1, phase1 = np.abs(F1), np.angle(F1)
mag2, phase2 = np.abs(F2), np.angle(F2)

img_m1p2 = np.abs(np.fft.ifft2(np.fft.ifftshift(mag1 * np.exp(1j * phase2))))
img_m2p1 = np.abs(np.fft.ifft2(np.fft.ifftshift(mag2 * np.exp(1j * phase1))))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes[0, 0].imshow(img1, cmap='gray');             axes[0, 0].set_title('Image 1 (Brick)')
axes[0, 1].imshow(img2, cmap='gray');             axes[0, 1].set_title('Image 2 (Grass)')
axes[0, 2].imshow(np.log1p(mag1), cmap='gray');   axes[0, 2].set_title('Magnitude 1 (log)')
axes[1, 0].imshow(img_m1p2, cmap='gray');         axes[1, 0].set_title('Mag1 + Phase2\n(Brick mag, Grass phase)')
axes[1, 1].imshow(img_m2p1, cmap='gray');         axes[1, 1].set_title('Mag2 + Phase1\n(Grass mag, Brick phase)')
axes[1, 2].imshow(np.log1p(mag2), cmap='gray');   axes[1, 2].set_title('Magnitude 2 (log)')
for ax in axes.ravel(): ax.axis('off')
plt.suptitle('Q3: Magnitude–Phase Swap Between Two Texture Images', fontsize=13)
plt.tight_layout()
plt.show()
