from google.colab import drive
drive.mount('/content/drive')
#將Colab與雲端硬碟連線
from google.colab.patches import cv2_imshow # 導入Colab.patches函式庫
import cv2 #導入opencv
img5 = cv2.imread('/content/drive/MyDrive/SummerCamp/Lenna.jpg')
detector = cv2.CascadeClassifier('/content/drive/MyDrive/SummerCamp/haarcascade_frontalface_default.xml') #讀取模型
gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
face = detector.detectMultiScale(gray, 1.1 ,3)
for(x, y, w, h) in face:
  cv2.rectangle(img5, (x,y), (x+w, y+h), (0,255,0),2)
cv2_imshow(img5)
