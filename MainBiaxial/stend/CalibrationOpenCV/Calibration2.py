import numpy as np
import cv2 as cv
import os
import glob
import timeit
import  re

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

def sort_by(file: str) -> int:
    # print( int(re.search(r'\d{3,}',file)[0]))
    return int(re.search(r'\d{3,}',file)[0])

def calibration(path,look_image = False ):
    print("Start calibration")
    folder_images = 'Calibration_images'
    folder_npy = 'Calibration_npy'



    if look_image:
        if not os.path.isdir(os.path.join(path,folder_images)):
            os.mkdir(os.path.join(path, folder_images))
        folder_location_image = os.path.join(path, folder_images)

    if not os.path.isdir(os.path.join(path, folder_npy)):
        os.mkdir(os.path.join(path, folder_npy))
    folder_location_npy = os.path.join(path, folder_npy)


    chessboardSize = (15,8)
    frameSize = (4096,3000)



    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 5
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('CalibrationOpenCV/*.jpg')
    # print(images)

    for image in images:
        # print(image)
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(1000)


    cv.destroyAllWindows()



    # print(imgpoints)
    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    print("Camera Calibrated: ", ret)
    print("\nCamera Matrix:\n", cameraMatrix)
    print("\nDistorsion Parameters:\n", dist)
    print("\nRotation Vectors:\n", rvecs)
    print("\nTranslation Vectors:\n", tvecs)

    ############## UNDISTORTION #####################################################
    frames = glob.glob(f'{path}/Frame/*.npy')
    j = 0
    frames.sort(key=sort_by)

    for frame in frames:
        start = timeit.default_timer()
        # img = cv.imread(frame)
        img = np.load(frame)
        h,  w = img.shape[:2]
        # print("h",h,"w",w)
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
        # print("roi",roi)
        #print("\nNewCamera Matrix:\n", newCameraMatrix)

        # Undistort
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        #cv.imwrite('caliResult3.jpg', dst)



        # Undistort with Remapping
        mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        # textlookfor = r'[0-9]+'
        # res = re.search(textlookfor, frame[-7:-3])
        # print(res[0])
        np.save(f'{folder_location_npy}/frame{j}', dst)
        if look_image:
            cv.imwrite(f'{folder_location_image}/frame{j}.tif', dst)

        end = timeit.default_timer()
        print(f"Time taken is colibration image {end - start}s")
        j += 1
    # Reprojection Error
    #     mean_error = 0
    #
    #     for i in range(len(objpoints)):
    #         imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    #         error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #         mean_error += error
    #
    #     print( "total error: {}".format(mean_error/len(objpoints)) )

    return folder_location_npy
