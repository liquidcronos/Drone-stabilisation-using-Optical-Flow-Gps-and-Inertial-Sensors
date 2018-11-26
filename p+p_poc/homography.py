import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

if __name__ == '__main__' :
 
    # Read source image.
    im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    pts_dst = np.array([[1765, 577], [2948, 632], [1298, 2084],[2849, 2282]])
 
 
    # Read destination image.
    im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_src = np.array([[2959, 363],[3861, 1292],[1441, 1226],[2249, 2409]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    #cv2.imshow("Source Image", im_src)
    #cv2.imshow("Destination Image", im_dst)
    #cv2.imshow("Warped Source Image", im_out)
    image=mpimg.imread("book1.jpg")
    plt.imshow(image)
    plt.quiver(pts_dst[:,0],pts_dst[:,1],pts_src[:,0]-pts_dst[:,0],pts_src[:,1]-pts_dst[:,1])

    plt.show()
    cv2.imwrite("Warped-Source.jpg",im_out)
 
    cv2.waitKey(0)
