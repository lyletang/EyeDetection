# coding:utf-8
# 静态图像中的人脸检测(包含眼检测)
# Author: Jiahui Tang

import cv2
import sys

imagePath = sys.argv[1]

def detect():
	faceCascade = cv2.CascadeClassifier("databases/haarcascade_frontalface_default.xml")
	eyeCascade = cv2.CascadeClassifier("databases/haarcascade_eye.xml")

	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.01,
		minNeighbors = 7,
		minSize = (30,30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
		
	for (x,y,w,h) in faces:
		cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

		#roi_gray = gray[y:y+h, x:x+w]

		
	eyes = eyeCascade.detectMultiScale(gray, 1.15, 6, 0, (10,10))

	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(image, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

	cv2.imshow("Found faces and eyes", image)
	cv2.imwrite("eyes.jpg", image)
	cv2.waitKey(0)
	#camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	detect()
