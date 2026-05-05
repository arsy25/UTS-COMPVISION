if 'gray_img' not in locals():
    print("Error: Citra grayscale (gray_img) tidak ditemukan. Pastikan Anda telah menjalankan sel sebelumnya.")
else:
    print("Menggunakan citra grayscale asli yang telah dianalisis sebelumnya.")

    # --- Penyesuaian Kecerahan dan Kontras pada Grayscale ---
    print("\n--- Menerapkan Penyesuaian Kecerahan dan Kontras pada Grayscale ---")
    alpha_gray = 1.5  # Faktor kontras
    beta_gray = 50    # Nilai kecerahan

    # Terapkan penyesuaian kecerahan dan kontras langsung pada citra grayscale
    adjusted_gray_img_new = cv2.convertScaleAbs(gray_img, alpha=alpha_gray, beta=beta_gray)

    print(f"Citra grayscale berhasil disesuaikan dengan kontras alpha={alpha_gray} dan kecerahan beta={beta_gray}.")

    # --- Tampilkan Citra Grayscale Asli dan yang Disesuaikan ---
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Citra Grayscale Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adjusted_gray_img_new, cmap='gray')
    plt.title(f'Citra Grayscale Disesuaikan (Kontras: {alpha_gray}, Kecerahan: {beta_gray})')
    plt.axis('off')

    plt.show()

    # --- Perbandingan Histogram Grayscale ---
    print("\n--- Perbandingan Histogram Citra Grayscale (Asli vs Disesuaikan) ---")
    # Hitung histogram untuk citra grayscale asli dan yang disesuaikan
    hist_gray_original_new = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_gray_adjusted_new = cv2.calcHist([adjusted_gray_img_new], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.plot(hist_gray_original_new, color='gray', label='Histogram Grayscale Asli')
    plt.plot(hist_gray_adjusted_new, color='blue', label='Histogram Grayscale Disesuaikan')
    plt.title('Perbandingan Histogram Citra Grayscale (Asli vs Disesuaikan)')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.legend()
    plt.grid(True)
    plt.show()
