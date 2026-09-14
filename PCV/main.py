import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_dirs = [
    os.path.join(BASE_DIR, "image"),
    os.path.join(BASE_DIR, "images")
]

IMAGE_DIR = next((d for d in candidate_dirs if os.path.isdir(d)), candidate_dirs[0])


def build_image_map():
    image_map = {}

    if not os.path.isdir(IMAGE_DIR):
        return image_map

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        stem = os.path.splitext(filename)[0].lower()
        label = {
            "terang": "Terang",
            "gelap": "Gelap",
            "kontras_rendah": "Kontras Rendah",
            "kontras rendah": "Kontras Rendah"
        }.get(stem, stem.replace("_", " ").title())

        image_map[label] = os.path.join(IMAGE_DIR, filename)

    return image_map


images = build_image_map()

def read_image(path):
    image = cv2.imread(path)

    if image is None:
        print(f"Gagal membaca gambar: {path}")
        return None

    return image


# ==========================================
# TUGAS 1
# Membaca dan menampilkan satu foto
# ==========================================

def tugas_1():
    print("\n========== TUGAS 1 ==========")

    if not images:
        print("Tidak ada gambar yang tersedia di folder citra.")
        return

    rows = 1
    cols = min(len(images), 3)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if cols == 1:
        axes = [axes]

    for ax, (nama, path) in zip(axes, images.items()):
        image = read_image(path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ax.imshow(gray, cmap='gray')
        ax.set_title(f"{nama} (Gray)")
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================
# TUGAS 2
# Shape, dtype, min, max, rata-rata
# ==========================================

def tugas_2():
    print("\n========== TUGAS 2 ==========")

    for nama, path in images.items():

        image = read_image(path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"\n--- {nama} ---")

        print("File       :", path)
        print("Shape      :", gray.shape)
        print("Dtype      :", gray.dtype)
        print("Nilai min  :", gray.min())
        print("Nilai max  :", gray.max())
        print("Rata-rata  :", gray.mean())


# ==========================================
# TUGAS 3
# Mencetak potongan 8 x 8 piksel
# ==========================================

def tugas_3():

    print("\n========== TUGAS 3 ==========")

    image_path = images.get("Terang") or next(iter(images.values()), None)

    if image_path is None:
        print("Tidak ada gambar yang tersedia untuk tugas 3.")
        return

    image = read_image(image_path)

    if image is None:
        return

    # Ubah menjadi grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    print("\nUkuran gambar grayscale:")
    print("Height :", height)
    print("Width  :", width)

    # ======================================
    # Potongan 8 x 8 dari area gelap
    # ======================================

    # Cari koordinat piksel paling gelap
    min_y, min_x = np.unravel_index(
        np.argmin(gray),
        gray.shape
    )

    # Pastikan koordinat tidak keluar batas
    dark_y = min(max(min_y - 4, 0), height - 8)
    dark_x = min(max(min_x - 4, 0), width - 8)

    dark_patch = gray[
        dark_y:dark_y + 8,
        dark_x:dark_x + 8
    ]

    # ======================================
    # Potongan 8 x 8 dari area terang
    # ======================================

    max_y, max_x = np.unravel_index(
        np.argmax(gray),
        gray.shape
    )

    bright_y = min(max(max_y - 4, 0), height - 8)
    bright_x = min(max(max_x - 4, 0), width - 8)

    bright_patch = gray[
        bright_y:bright_y + 8,
        bright_x:bright_x + 8
    ]

    # ======================================
    # PRINT
    # ======================================

    print("\nPotongan 8 x 8 AREA GELAP:")
    print(dark_patch)

    print("\nRentang nilai area gelap:")
    print(
        dark_patch.min(),
        "sampai",
        dark_patch.max()
    )

    print("\nPotongan 8 x 8 AREA TERANG:")
    print(bright_patch)

    print("\nRentang nilai area terang:")
    print(
        bright_patch.min(),
        "sampai",
        bright_patch.max()
    )


# ==========================================
# TUGAS 4
# Ukuran data mentah dan rasio kompresi
# ==========================================

def tugas_4():

    print("\n========== TUGAS 4 ==========")

    path = images.get("Terang") or next(iter(images.values()), None)

    if path is None:
        print("Tidak ada gambar yang tersedia untuk tugas 4.")
        return

    image = read_image(path)

    if image is None:
        return

    # Shape gambar
    height, width, channels = image.shape

    # dtype biasanya uint8
    bytes_per_pixel = image.dtype.itemsize

    # Ukuran data mentah
    raw_size = height * width * channels * bytes_per_pixel

    # Ukuran file sebenarnya
    file_size = os.path.getsize(path)

    # Rasio kompresi
    compression_ratio = raw_size / file_size

    print("\nInformasi gambar:")
    print("Nama file          :", path)
    print("Height             :", height)
    print("Width              :", width)
    print("Jumlah channel     :", channels)
    print("Dtype              :", image.dtype)
    print("Ukuran 1 data      :", bytes_per_pixel, "byte")

    print("\nUkuran data mentah:")
    print("=", height, "x", width, "x", channels, "x", bytes_per_pixel)
    print("=", raw_size, "byte")

    print("\nUkuran file:")
    print("=", file_size, "byte")

    print("\nRasio kompresi:")
    print("=", raw_size, "/", file_size)
    print("=", compression_ratio)

    print(
        f"\nJadi rasio kompresinya sekitar "
        f"{compression_ratio:.2f}:1"
    )


# ==========================================
# PROGRAM UTAMA
# ==========================================

def main():

    print("======================================")
    print("       TUGAS PENGOLAHAN CITRA")
    print("======================================")

    tugas_1()
    tugas_2()
    tugas_3()
    tugas_4()


if __name__ == "__main__":
    main()