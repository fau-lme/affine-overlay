"""
copyright: Vincent Christlein, vincent.christlein@fau.de
licence: Apache License v 2.0, see LICENSE
This program overlays/blends two images (or their contours) with each other. 
    For this process first three (affine) or more points (perspective) have to
    be selected at the two images (note: the point order must match). With
    <space> you continue to the next view. At the end you see the result, if you
    want to redo: press <r>. 
"""

import numpy as np
import sys
import os
import glob
import argparse
import cv2

def getPoints(img):     
    points = []
    def onmouse2(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
            cv2.circle(img, (x,y), 3, (255,0,0))
            cv2.imshow("img", img)

    cv2.namedWindow("img", 1)
    cv2.setMouseCallback("img", onmouse2)
    while True:
        cv2.imshow("img", img)
        # space to exit this window
        key = cv2.waitKey() & 255
        if key == ord(' ') or key == 27:
            cv2.destroyAllWindows()
            return points

# somehow these need to be global
drag_start = None
sel = (0,0,0,0)
done = None
def getSelection(img):
    def onmouse(event, x, y, flags, param):
        global drag_start, sel
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = x, y
            sel = 0,0,0,0
        elif event == cv2.EVENT_LBUTTONUP:
            drag_start = None
        elif drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                minpos = min(drag_start[0], x), min(drag_start[1], y)
                maxpos = max(drag_start[0], x), max(drag_start[1], y)
                sel = minpos[0], minpos[1], maxpos[0], maxpos[1]
                tmp = img.copy()
                cv2.rectangle(tmp, (sel[0], sel[1]), (sel[2], sel[3]), (0,255,255), 1)
                cv2.imshow("select", tmp)
                
    cv2.namedWindow("select", 1)
    cv2.setMouseCallback("select", onmouse)
    cv2.imshow("select", img)
    if (cv2.waitKey() & 255) == 27:
        cv2.destroyAllWindows()
        return sel

def selectResize(img, width=800):
    factor = img.shape[1] / float(width)
    r1 = cv2.resize(img, (width, int(img.shape[0] / factor)) )    
    roi1 = np.array( getSelection(r1) )
    roi1 = roi1 * factor
    patch = img[roi1[1]:roi1[3],roi1[0]:roi1[2]]
    factor = patch.shape[1] / float(width)
    img = cv2.resize(patch, (width, int(patch.shape[0] / factor)) )
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('image warping')
    parser.add_argument('img1', help='path to first image')
    parser.add_argument('img2', help='path to second image')
    parser.add_argument('-w', '--width', type=int, default=800, 
                        help='scale image to that width')
    parser.add_argument('--blur-contour', action='store_true', 
                        help='blur the contour')
    parser.add_argument('--eksize', type=int, default=0,
                        help='erosion kernel size')
    parser.add_argument('--save_result', action='store_true',
                        help='save intermediate results as pngs in the local'
                        ' folder')
    args = parser.parse_args()

    im1 = cv2.imread(args.img1) #, cv2.IMREAD_GRAYSCALE)
    im1 = selectResize(im1, args.width)    
    im2 = cv2.imread(args.img2) #, cv2.IMREAD_GRAYSCALE)
    drag_start = None
    sel = (0,0,0,0)
    done = None
    im2 = selectResize(im2, args.width)
   
    it = 0
   
    while True:
        p1 = getPoints(im1.copy())
        p2 = getPoints(im2.copy())

        assert(len(p1) == len(p2))
        assert(len(p1) > 2)

        p1 = np.array(p1, np.float32)
        p2 = np.array(p2, np.float32)
        if len(p1) == 3: 
            A = cv2.getAffineTransform(p1, p2)
            warped = cv2.warpAffine(im1, A, (im2.shape[1], im2.shape[0]))
        else:
            if len(p1) == 4:
                A = cv2.getPerspectiveTransform(p1, p2)
            else:
                A = cv2.findHomography(p1, p2)
            warped = cv2.warpPerspective(im1, A, (im2.shape[1], im2.shape[0]))

        def modifyImg(img, blur, contour, blur_contour=False, eksize=0):
            if blur > 0:
                t1 = cv2.GaussianBlur(img , (0,0), blur)
            else:
                t1 = img.copy()
            if contour > 0:
                t1 = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
                t1 = cv2.Canny(t1, 50, 100)
                t1 = np.invert(t1)
                if eksize > 1:
                    t1 = cv2.erode(t1, np.ones( (eksize,eksize), dtype=np.int32))
                if blur_contour:
                    t1 = cv2.GaussianBlur(t1, (0,0), 2.0)
                t1 = cv2.cvtColor(t1, cv2.COLOR_GRAY2BGR)

            return t1

        def update(dummy=None):       
            global it
            b1 = cv2.getTrackbarPos('blur-radius1', 'result2') / 10.0
            c1 = cv2.getTrackbarPos('contour1', 'result2')
            t1 = modifyImg(warped, b1, c1, args.blur_contour, args.eksize)
            if c1 > 0: # colorize contour
                green = t1[:,:,1]
                green[ green != 255 ] = 139
                blue = t1[:,:,2]
                blue[ blue != 255 ] = 255

            b2 = cv2.getTrackbarPos('blur-radius2', 'result2') / 10.0
            c2 = cv2.getTrackbarPos('contour2', 'result2')
            t2 = modifyImg(im2, b2, c2, args.blur_contour, args.eksize)
            if c2 > 0: # colorize contour 
                red = t2[:,:,0]
                red[ red != 255 ] = 200
                green = t2[:,:,1]
                green[ green != 255 ] = 137
            
            # blend images
            bf = cv2.getTrackbarPos('blend', 'result2') / 100.0        
            dst = cv2.addWeighted(t1, bf, t2, 1.0 - bf, 0)
            cv2.imshow("result", dst)
            if args.save_result:
                cv2.imwrite('result_{}.png'.format(it), dst)
                it += 1
        
        cv2.namedWindow("result2")
        cv2.createTrackbar('blend', 'result2', 60, 100, update)
        cv2.createTrackbar('blur-radius1', 'result2', 1, 200, update)
        cv2.createTrackbar('blur-radius2', 'result2', 1, 200, update)
        cv2.createTrackbar('contour1', 'result2', 0, 1, update)
        cv2.createTrackbar('contour2', 'result2', 0, 1, update)
        while True:
            update() 
            key = cv2.waitKey()
            if (key & 255) == 27:
                cv2.destroyAllWindows()
                sys.exit()
            # redo the selection part
            elif (key & 255) == ord('r'):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

