import cv2
import numpy as np
image = cv2.imread("night.jpg")
if image is None:
    print("Error: night.jpg not found!")
    exit()
image = cv2.resize(image, (400, 300))
cv2.imwrite("1_original.jpg", image)
gamma = 2.0
lookup_table = np.array(
    [((i / 255.0) ** (1 / gamma)) * 255 for i in range(256)],
    dtype="uint8"
)
gamma_corrected = cv2.LUT(image, lookup_table)
cv2.imwrite("2_gamma.jpg", gamma_corrected)
lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)
l = clahe.apply(l)
lab = cv2.merge((l, a, b))
clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
cv2.imwrite("3_clahe.jpg", clahe_image)
denoised = cv2.GaussianBlur(clahe_image, (5, 5), 0)
cv2.imwrite("4_denoised.jpg", denoised)
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = cv2.filter2D(denoised, -1, kernel)
cv2.imwrite("5_sharpened.jpg", sharpened)
gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
_, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
cv2.imwrite("6_otsu.jpg", otsu)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite("7_edges.jpg", edges)
cv2.imshow("1. Original", image)
cv2.imshow("2. Gamma Corrected", gamma_corrected)
cv2.imshow("3. CLAHE", clahe_image)
cv2.imshow("4. Denoised", denoised)
cv2.imshow("5. Sharpened", sharpened)
cv2.imshow("6. Otsu Segmentation", otsu)
cv2.imshow("7. Canny Edge Detection", edges)
print("===================================")
print("Processing Completed Successfully!")
print("Output Images Saved:")
print("1_original.jpg")
print("2_gamma.jpg")
print("3_clahe.jpg")
print("4_denoised.jpg")
print("5_sharpened.jpg")
print("6_otsu.jpg")
print("7_edges.jpg")
print("===================================")
cv2.waitKey(0)
cv2.destroyAllWindows()
