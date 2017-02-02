# coding:utf-8
# 视频中的人脸检测(包含眼检测)
# Author: Jiahui Tang

import cv2
import sys

def detect():
	faceCascade = cv2.CascadeClassifier("databases/haarcascade_frontalface_default.xml")
	eyeCascade = cv2.CascadeClassifier("databases/haarcascade_eye.xml")

	camera = cv2.VideoCapture('2.mp4')

	while (True):
		ret, frame = camera.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor = 1.06,
			minNeighbors = 7,
			minSize = (30,30),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		
		for (x,y,w,h) in faces:
			img = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

			roi_gray = gray[y:y+h, x:x+w]

			eyes = eyeCascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40,40))

			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

		cv2.imshow("camera", frame)
		if cv2.waitKey(1000/12) & 0xFF == ord("q"):
			break

	#camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	detect()
