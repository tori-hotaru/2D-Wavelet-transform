import pywt
import numpy as np
import cv2
import os

# Load the image
img_path = "/Your project path"
img = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply wavelet transform
coeffs = pywt.dwt2(gray, 'haar')# coeffs是一个tuple，tuple里面有两个元素，第一个元素是一个numpy.ndarray，第二个元素也是一个numpy.ndarray，haar代表haar小波
cA, (cH, cV, cD) = coeffs # cA: approximation, cH: horizontal detail, cV: vertical detail, cD: diagonal detail
# cA, (cH, cV, cD)代表的是一个tuple，cA是一个numpy.ndarray，(cH, cV, cD)是一个tuple，tuple里面的每一个元素都是一个numpy.ndarray

# 将不进行小波逆变换的图像保存在原图位置的文件夹里
# Save the image
output_path = os.path.join(os.path.dirname(img_path), "photo_chang_cA.jpg") # os.path.dirname()函数的作用是返回文件路径的目录，join()函数的作用是将多个路径组合后返回
cv2.imwrite(output_path, cA) #imwrite()函数的第一个参数是保存的路径，第二个参数是要保存的图像

output_path = os.path.join(os.path.dirname(img_path), "photo_change_cD.jpg") # os.path.dirname()函数的作用是返回文件路径的目录，join()函数的作用是将多个路径组合后返回
cv2.imwrite(output_path, cD) #imwrite()函数的第一个参数是保存的路径，第二个参数是要保存的图像

output_path = os.path.join(os.path.dirname(img_path), "photo_change_cH.jpg") # os.path.dirname()函数的作用是返回文件路径的目录，join()函数的作用是将多个路径组合后返回
cv2.imwrite(output_path, cH) #imwrite()函数的第一个参数是保存的路径，第二个参数是要保存的图像

output_path = os.path.join(os.path.dirname(img_path), "photo_change_cV.jpg") # os.path.dirname()函数的作用是返回文件路径的目录，join()函数的作用是将多个路径组合后返回
cv2.imwrite(output_path, cV) #imwrite()函数的第一个参数是保存的路径，第二个参数是要保存的图像

# Reconstruct the image using inverse wavelet transform
reconstructed_img = pywt.idwt2(coeffs, 'haar')

# Save the reconstructed image
output_path = os.path.join(os.path.dirname(img_path), "reconstructed_photo.jpg")
cv2.imwrite(output_path, reconstructed_img)

