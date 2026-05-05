import cv2
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

# 1. Upload citra
print("Silakan unggah citra Anda. Pastikan berukuran minimal 512x512.")
uploaded = files.upload()

image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)

# Periksa apakah citra berhasil dimuat
if img is None:
    print(f"Error: Tidak dapat memuat citra dari {image_path}")
else:
    print(f"Citra '{image_path}' berhasil dimuat.")

    # 2. Tampilkan Dimensi citra
    height, width, channels = img.shape
    print(f"\nDimensi Citra (Tinggi x Lebar): {height} x {width} piksel")

    # 3. Jumlah channel
    print(f"Jumlah Channel: {channels}")

    # 4. Resolusi (jumlah total piksel)
    resolution = height * width
    print(f"Resolusi Citra: {resolution} piksel ({width}x{height})")

    # 5. Konversi ke grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Citra berhasil dikonversi ke skala abu-abu.")

    # 6. Tampilkan nilai pixel pada koordinat tertentu (misal (100,100))
    # Ingat, koordinat (y, x) untuk NumPy/OpenCV
    x_coord, y_coord = 100, 100

    if y_coord < height and x_coord < width:
        pixel_value_color = img[y_coord, x_coord]
        pixel_value_gray = gray_img[y_coord, x_coord]
        print(f"\nNilai piksel pada koordinat ({x_coord},{y_coord}) di citra asli (BGR): {pixel_value_color}")
        print(f"Nilai piksel pada koordinat ({x_coord},{y_coord}) di citra grayscale: {pixel_value_gray}")
    else:
        print(f"Koordinat ({x_coord},{y_coord}) berada di luar batas citra.")

    # Tampilkan citra asli dan grayscale
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for matplotlib
    plt.title('Citra Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale')
    plt.axis('off')

    plt.show()
