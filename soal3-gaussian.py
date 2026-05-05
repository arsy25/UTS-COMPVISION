# Pastikan variabel gray_img tersedia dari eksekusi sebelumnya
if 'gray_img' not in locals():
    print("Error: Citra grayscale (gray_img) tidak ditemukan. Pastikan Anda telah menjalankan sel sebelumnya.")
else:
    print("Melanjutkan operasi pada citra grayscale yang sudah ada.")

    # --- Gaussian Blur ---
    print("\n--- Menerapkan Gaussian Blur (kernel 5x5) ---")
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian_blur, cmap='gray')
    plt.title('Gaussian Blur (Kernel 5x5)')
    plt.axis('off')
    plt.show()

    # --- Median Blur ---
    print("\n--- Menerapkan Median Blur (kernel 5x5) ---")
    median_blur = cv2.medianBlur(gray_img, 5)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(median_blur, cmap='gray')
    plt.title('Median Blur (Kernel 5x5)')
    plt.axis('off')
    plt.show()
