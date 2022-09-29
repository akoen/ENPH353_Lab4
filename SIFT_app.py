#!/usr/bin/env python3

from re import search
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import numpy as np
import cv2 as cv
import sys
import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class My_App(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./L04/SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 30
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self.load_reference('./L04/robot.jpg')
        self.sift = cv.SIFT_create()

        self._camera_device = cv.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(int(1000 / self._cam_fps))

        self.SLOT_toggle_camera()

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        self.load_reference(self.template_path)

    def load_reference(self, path):
        self.img_ref = cv.imread(path)
        self.template_label.setPixmap(self.convert_cv_to_pixmap(self.img_ref))
    
    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, img_cam = self._camera_device.read()
        gray_ref = cv.cvtColor(self.img_ref, cv.COLOR_BGR2GRAY)
        gray_cam = cv.cvtColor(img_cam, cv.COLOR_BGR2GRAY)

        kp_ref, des_ref = self.sift.detectAndCompute(gray_ref, None)
        kp_cam, des_cam = self.sift.detectAndCompute(gray_cam, None)

        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_ref, des_cam, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>10:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp_cam[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h,w = self.img_ref.shape[0:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)

            img_cam = cv.polylines(img_cam,[np.int32(dst)], True, 255, 3, cv.LINE_AA)

        else:
            matchesMask = None
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        img = cv.drawMatches(self.img_ref,kp_ref,img_cam,kp_cam,good,None,**draw_params)

        pixmap = self.convert_cv_to_pixmap(img)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
