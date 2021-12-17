import numpy as np
import cv2
import glob
import os


#import pathlab

corner_x=10 # number of chessboard corner in x direction
corner_y=7 # number of chessboard corner in y direction

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corner_y*corner_x,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x,0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points.
objpoints = [] # 3d point in real world space
jpegpoints = [] # 2d points in image plane.


images =glob.glob('./Data/*.jpeg')



found = 0
for fname in images: # here, 10 can be changed to whatever number you like to choose
    jpeg = cv2.imread(fname) # capture frame by frame
    jpeg=cv2.resize(jpeg,(700,700))
    cv2.imshow('jpeg', jpeg)
    cv2.waitKey(50)
    print(fname)
    gray = cv2.cvtColor(jpeg, cv2.COLOR_BGRA2GRAY)
    
    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
    # if found, pass object points, image points (after refining them)
    if ret == True:
        
        objpoints.append(objp) #Certainly, every loop objp is the same in 3D
        corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        jpegpoints.append(corners2)
        # Draw and display the corners
        jpeg = cv2.drawChessboardCorners(jpeg, (corner_x,corner_y), corners2, ret)
        found += 1
        jpeg=cv2.resize(jpeg,(700,700))
        cv2.imshow('chessboard', jpeg)
        cv2.waitKey(0)
        # if you want to save images with detected corners
      
        
print("number of images used for calibration: ", found)



#calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, jpegpoints, gray.shape[::-1], None, None)

# transforms the matrix distortion coefficients to writeable lists
data= {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
print("Intrinsic Camera matrix:\n",mtx)
print("Distortion Coefficients:\n",dist)

     
cv2.destroyAllWindows() 
