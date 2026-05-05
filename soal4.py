import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from google.colab import files
import os # Import the os module

uploaded = files.upload()
img_path = next(iter(uploaded.keys()))

img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise ValueError('Gagal membaca gambar')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
edges = cv2.Canny(img_blur, 50, 150)

_, seg = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.mean(seg) > 127:
    seg = cv2.bitwise_not(seg)

kernel = np.ones((3, 3), np.uint8)
seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel, iterations=1)
seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel, iterations=2)

hog_feat, hog_vis = hog(
    img_blur,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    feature_vector=True
)
hog_vis_rescaled = exposure.rescale_intensity(hog_vis, in_range=(0, 10))

out_dir = 'output'

# Create the output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(16, 10))
figs = [
    ('1. Original', img_rgb, 'rgb'),
    ('2. Grayscale', img_gray, 'gray'),
    ('3. Blur', img_blur, 'gray'),
    ('4. Edges', edges, 'gray'),
    ('5. Segmentasi', seg, 'gray'),
    ('6. HOG', hog_vis_rescaled, 'gray'),
]

for i, (title, im, cmap) in enumerate(figs, 1):
    ax = plt.subplot(2, 3, i)
    ax.imshow(im if cmap == 'rgb' else im, cmap=None if cmap == 'rgb' else 'gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{out_dir}/pipeline_hasil.png', dpi=200, bbox_inches='tight')
plt.close()

flow = '[Load Citra] -> [Grayscale + Blur] -> [Edge Detection] -> [Segmentasi] -> [Ekstraksi Fitur HOG/SIFT]'
with open(f'{out_dir}/diagram_alur.txt', 'w', encoding='utf-8') as f:
    f.write(flow)

insight = (
    'Tahap paling berpengaruh biasanya preprocessing + segmentasi, karena kualitas edge dan fitur sangat '
    'dipengaruhi oleh noise, kontras, dan apakah objek berhasil dipisahkan dari background. Jika segmentasi '
    'buruk, HOG/SIFT cenderung menangkap banyak informasi latar.'
)
with open(f'{out_dir}/insight.txt', 'w', encoding='utf-8') as f:
    f.write(insight)

print('Selesai')
