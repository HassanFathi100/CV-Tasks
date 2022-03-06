"""Functions that may help generally"""
import cv2

#function to store an img to local hard
def store_img(name, img):
    cv2.imwrite(name, img)
