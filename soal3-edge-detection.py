# Pastikan variabel gray_img tersedia
if 'gray_img' not in locals():
    print("Error: Citra grayscale (gray_img) tidak ditemukan.")
else:
    # --- Canny Edge Detection ---
    print("\n--- Melakukan Deteksi Tepi Canny ---")
    # Menggunakan nilai threshold umum, bisa disesuaikan
    canny_edges = cv2.Canny(gray_img, 100, 200)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    plt.show()

    # --- Sobel Edge Detection ---
    print("\n--- Melakukan Deteksi Tepi Sobel ---")
    # Menghitung gradien X dan Y
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)

    # Menggabungkan gradien X dan Y
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Normalisasi untuk tampilan
    sobel_display = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_display, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    plt.show()
