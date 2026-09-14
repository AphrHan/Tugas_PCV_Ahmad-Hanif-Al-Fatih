import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/terang.jpg")

if image is None:
    print("Gambar tidak ditemukan!")
    exit()

blue, green, red = cv2.split(image)

red_image = np.zeros_like(image)
green_image = np.zeros_like(image)
blue_image = np.zeros_like(image)

red_image[:, :, 2] = red
green_image[:, :, 1] = green
blue_image[:, :, 0] = blue

original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
red_rgb = cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB)
green_rgb = cv2.cvtColor(green_image, cv2.COLOR_BGR2RGB)
blue_rgb = cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(red_rgb)
plt.title("Red")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(green_rgb)
plt.title("Green")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(blue_rgb)
plt.title("Blue")
plt.axis("off")

plt.tight_layout()
plt.show()
