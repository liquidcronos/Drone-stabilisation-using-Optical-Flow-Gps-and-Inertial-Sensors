import numpy as np
import cv2
data=np.load("pic2.txt.npy")
np.set_printoptions(threshold=np.nan)
print(data.shape)
print(data)
cv2.imshow("test",data)
