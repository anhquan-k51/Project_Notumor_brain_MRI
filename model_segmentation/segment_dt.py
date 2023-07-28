import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
import skimage.transform as trans
import sys
sys.path.insert(1, 'G:/Vlancer_Project/157.Nhan_dang_Phan_loai_U_nao/segmentation/ImageSegmentation/')
import model_bt

image_path = 'G:/Vlancer_Project/157.Nhan_dang_Phan_loai_U_nao/segmentation/imgForTest/t (3).jpg'
unet_weight = 'G:/Vlancer_Project/157.Nhan_dang_Phan_loai_U_nao/segmentation/model_x/segmentation/weights.hdf5'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image / 255
image = trans.resize(image,(256,256))

org = cv2.imread(image_path)
rows, cols, channels = org.shape     
unet = model_bt.unet_bt(pretrained_weights=unet_weight)
unet.compile(optimizer = Adam(), loss = 'binary_crossentropy')
predicted = unet.predict(np.reshape(image, (1, 256, 256, 1)))
predicted = predicted.astype(np.float64) * 255
predicted = np.reshape(predicted, (256, 256))
predicted = trans.resize(predicted, (rows,cols))
predicted = predicted.astype(np.uint8)
predicted = cv2.cvtColor(predicted, cv2.COLOR_GRAY2BGR)

ret, mask = cv2.threshold(predicted, 120, 255, cv2.THRESH_BINARY)
white_pixels = np.where((mask[:, :, 0] == 255) & 
                        (mask[:, :, 1] == 255) & 
                        (mask[:, :, 2] == 255))
mask[white_pixels] = [0, 0, 255]
add = cv2.addWeighted(org, 0.9, mask, 0.7, 0)   

cv2.imshow('image', add)
cv2.waitKey(0)
cv2.destroyAllWindows()