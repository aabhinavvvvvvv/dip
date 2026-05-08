import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.data import brick, grass, gravel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def get_patches(texture_img, n=20, size=64):
    h, w = texture_img.shape
    patches = []
    for _ in range(n):
        r = np.random.randint(0, h - size)
        c = np.random.randint(0, w - size)
        patches.append(texture_img[r:r+size, c:c+size])
    return patches

def extract_glcm_features(patch):
    img = (patch * 255).astype(np.uint8) if patch.max() <= 1.0 else patch.astype(np.uint8)
    glcm = graycomatrix(img, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'energy', 'homogeneity', 'correlation']:
        feats.extend(graycoprops(glcm, prop).flatten())
    return feats

# ── Dataset: 3 built-in texture classes, 20 patches each ──────────────────
textures = [brick(), grass(), gravel()]
class_names = ['brick', 'grass', 'gravel']

# Show texture classes
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, tex, name in zip(axes, textures, class_names):
    ax.imshow(tex, cmap='gray')
    ax.set_title(name)
    ax.axis('off')
plt.suptitle('Texture Classes', fontsize=13)
plt.tight_layout()
plt.show()

# ── Extract GLCM features ──────────────────────────────────────────────────
X, y = [], []
for label, tex in enumerate(textures):
    for patch in get_patches(tex, n=20, size=64):
        X.append(extract_glcm_features(patch))
        y.append(label)

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ── Train classifiers ──────────────────────────────────────────────────────
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")

# ── Confusion matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, title in zip(axes, [y_pred_svm, y_pred_knn], ['SVM', 'KNN']):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.set_title(f'{title} Confusion Matrix\n(Accuracy: {accuracy_score(y_test, pred):.4f})')
plt.suptitle('GLCM-Based Texture Classification', fontsize=14)
plt.tight_layout()
plt.show()
