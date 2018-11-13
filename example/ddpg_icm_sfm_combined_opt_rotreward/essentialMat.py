import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def getMatrices(img1, img2):

    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    cv2.ocl.setUseOpenCL(False)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F= 0
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    E, mask = cv2.findEssentialMat(pts1,pts2) #https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2)
    #print("Fundamental Mat:\n", F)
    #print("Essential Mat:\n", E)
    #print(E)
    #print(R)
    #print(t)
    ''
    ''' Plotting '''
    '''
    # We select only inlier points
    #pts1 = pts1[mask.ravel()==1]
    #pts2 = pts2[mask.ravel()==1]

    print(pts1,pts2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    plt.close()
    '''
    return F,E, R, t

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :


    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

if __name__ == '__main__':
    left_fn = '/hdd/AIRSCAN/sfm_results/icm_fixed_lr_circ/images/ob0.png'
    right_fn = '/hdd/AIRSCAN/sfm_results/icm_fixed_lr_circ/images/ob1.png'
    img1 = cv2.imread(left_fn,0)  #queryimage # left image
    # img1 = cv2.imread(left_fn,cv2.IMREAD_COLOR)  #queryimage # left image
    print('shape image: ', img1.shape)
    img2 = cv2.imread(right_fn,0) #trainimage # right image
    # img2 = cv2.imread(right_fn,cv2.IMREAD_COLOR) #trainimage # right image
    F,E, R, t = getMatrices(img1,img2)
    print('R: ', R)
    print('R shape: ', R.shape)
    print('t: ', t)


    Rx = 0
    Ry = 30
    Rz = 30

    theta = [0, math.radians(Ry), math.radians(Rz)]
    R = eulerAnglesToRotationMatrix(theta)
    print(R)
