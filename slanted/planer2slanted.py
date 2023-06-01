import cv2
import numpy as np

# 入力画像の読み込み
img = cv2.imread('input_image.jpg')

# 変換前の座標を設定
#"1000,148,197,1860,3752,2560,3730,420,3729,423" h
pts1 = np.float32([[1000,148],[197,1860],[3752,2560],[3729,423]])

# 変換後の座標を設定
width = 4032
height = 3024
pts2 = np.float32([[0,0],[0,height],[width,height],[width,0]])

# 透視変換行列を計算
M = cv2.getPerspectiveTransform(pts1, pts2)

# 透視変換を実行
dst = cv2.warpPerspective(img, M, (width, height))

# 結果を表示
cv2.imwrite('output_image.jpg', dst)
