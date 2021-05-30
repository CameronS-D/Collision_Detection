import cv2
import numpy as np

class Image:
    margin_sf = 0.2

    def __init__(self, img, bgs):
        self.original = img
        self.bgs = bgs

        self.h, self.w = img.shape[:2]
        self.xmargin = int( Image.margin_sf * self.w )
        self.ymargin = int( Image.margin_sf * self.h )

        self.min_contour_area = 0.01 * self.h

        self.cropped = img[self.ymargin:-self.ymargin, self.xmargin:-self.xmargin]
        self.grey = self.get_grey_contour_img()

    def get_grey_contour_img(self):

        # get foreground mask from image and find contours in it
        fgMask = self.bgs.apply(self.cropped, learningRate=0.1)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # eliminate small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]

        # add contours to grey image
        tempImg = np.zeros((*self.cropped.shape[:2], 1), np.uint8)
        cv2.drawContours(tempImg, contours, -1, 255, -1)

        return tempImg

    def show(self, keyPoints, is_foreground=None, title="Image", vid_writer=None):

        if is_foreground is None:
            is_foreground = [False] * len(keyPoints)

        temp_img = self.original
        # add keypoints to image, then save or show user
        for coord, near_front in zip(keyPoints, is_foreground):
            x, y = coord.ravel()
            centre = ( int(x) + self.xmargin, int(y) + self.ymargin )

            # colour in BGR
            colour = (255, 0, 0)
            if near_front:
                colour = (0, 0, 255)

            cv2.circle(temp_img, centre, 1, colour, 2)

        if vid_writer is not None:
            vid_writer.write(temp_img)
        else:
            cv2.imshow(title, temp_img)
            cv2.waitKey(1)
