if 'img' not in locals() or 'gray_img' not in locals():
    print("Error: Citra asli (img) atau citra grayscale (gray_img) tidak ditemukan. Pastikan Anda telah menjalankan sel sebelumnya.")
else:
    print("Menggunakan citra asli yang telah diunggah dan dianalisis sebelumnya.")

    # --- Penyesuaian Kecerahan dan Kontras ---
    print("\n--- Menerapkan Penyesuaian Kecerahan dan Kontras ---")
    alpha = 1.5  # Faktor kontras (dari permintaan: alpha = 1.5)
    beta = 50    # Nilai kecerahan (dari permintaan: Brightness +50)

    # Terapkan penyesuaian kecerahan dan kontras
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    print(f"Citra berhasil disesuaikan dengan kontras alpha={alpha} dan kecerahan beta={beta}.")

    # Konversi citra yang telah disesuaikan ke grayscale untuk histogram
    adjusted_gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)

    # --- Tampilkan Citra Asli dan yang Disesuaikan ---
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Citra Disesuaikan (Kontras: {alpha}, Kecerahan: {beta})')
    plt.axis('off')

    plt.show()

    # --- Perbandingan Histogram ---
    print("\n--- Perbandingan Histogram ---")
    # Hitung histogram untuk citra grayscale asli dan yang disesuaikan
    hist_gray_original = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_gray_adjusted = cv2.calcHist([adjusted_gray_img], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.plot(hist_gray_original, color='gray', label='Histogram Grayscale Asli')
    plt.plot(hist_gray_adjusted, color='blue', label='Histogram Grayscale Disesuaikan')
    plt.title('Perbandingan Histogram Citra Grayscale (Asli vs Disesuaikan)')
    plt.xlabel('Intensitas Piksel')
    plt.ylabel('Frekuensi')
    plt.legend()
    plt.grid(True)
    plt.show()
