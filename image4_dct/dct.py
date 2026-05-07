import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from skimage.data import camera
from skimage import img_as_float

def dct2(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

# ── Q1: 2D DCT — visualize coefficients and IDCT reconstruction ───────────
img = img_as_float(camera())

dct_coeffs  = dct2(img)
img_recon   = idct2(dct_coeffs)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray');                              axes[0].set_title('a) Original Image')
axes[1].imshow(np.log1p(np.abs(dct_coeffs)), cmap='hot');     axes[1].set_title('b) DCT Coefficients (log-scaled)')
axes[2].imshow(img_recon, cmap='gray');                        axes[2].set_title('c) IDCT Reconstructed Image')
for ax in axes: ax.axis('off')
plt.suptitle('Q1: 2D DCT Visualization', fontsize=14)
plt.tight_layout()
plt.show()
print(f"Q1 Max reconstruction error: {np.max(np.abs(img - img_recon)):.2e}")

# ── Q2: DCT Basis Images for 8×8 and 16×16 ────────────────────────────────
def dct_basis_images(N):
    basis = np.zeros((N, N, N, N))
    for u in range(N):
        for v in range(N):
            delta = np.zeros((N, N))
            delta[u, v] = 1.0
            basis[u, v] = idct(idct(delta.T, norm='ortho').T, norm='ortho')
    return basis

for N in [8, 16]:
    basis = dct_basis_images(N)
    vmax  = np.max(np.abs(basis))

    fig, axes = plt.subplots(N, N, figsize=(N * 0.9, N * 0.9))
    fig.suptitle(f'Q2: DCT Basis Images ({N}×{N})', fontsize=12)
    for u in range(N):
        for v in range(N):
            axes[u, v].imshow(basis[u, v], cmap='gray', vmin=-vmax, vmax=vmax)
            axes[u, v].axis('off')
    plt.tight_layout()
    plt.show()
