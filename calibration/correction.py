import cv2
import numpy as np

# カメラ行列と歪みパラメータを読み込み
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

# 入力画像の読み込み
img = cv2.imread('input_image.jpg')

# カメラ行列と歪みパラメータを用いて画像補正
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# ROIで画像を切り抜く
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]

# 補正後の画像を表
# save_img = "calibresult.png"
save_img = "output_image.jpg"
cv2.imwrite(save_img, dst)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
