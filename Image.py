import cv2
import numpy as np

class Image:
    margin_sf = 0.2

    def __init__(self, img, bgs):
        # resize image to lower resolution to reduce computation cost
        self.original = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
        self.bgs = bgs

        self.h, self.w = self.original.shape[:2]
        self.xmargin = int( Image.margin_sf * self.w )
        self.ymargin = int( Image.margin_sf * self.h )

        self.min_contour_area = 0.01 * self.h

        self.cropped = self.original[self.ymargin:-self.ymargin, self.xmargin:-self.xmargin]
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

    def add_features(self, obstacles=None, keypoints=None, is_foreground=None):

        # add obstacles
        # colours in BGR
        if obstacles is not None:
            centroids = obstacles["centroids"]
            dims = obstacles["dims"]

            for (x, y), (w, h) in zip(centroids, dims):
                start_cnr = (   self.xmargin + x - w//2,
                                self.ymargin + y - h//2)
                end_cnr = ( self.xmargin + x + w//2,
                            self.ymargin + y + h//2)

                cv2.rectangle(self.original, start_cnr, end_cnr, color=(0, 0, 255), thickness=5)

        # add keypoints
        if keypoints is not None:
            if is_foreground is None:
                is_foreground = [False] * len(keypoints)

            for coord, near_front in zip(keypoints, is_foreground):
                x, y = coord.ravel()
                centre = ( int(x) + self.xmargin, int(y) + self.ymargin )

                # colour in BGR
                colour = (255, 0, 0)
                if near_front:
                    colour = (0, 0, 255)

                cv2.circle(self.original, centre, radius=2, color=colour, thickness=-1)



    def show(self, title="Image", vid_writer=None):

        # if is_foreground is None:
        #     is_foreground = [False] * len(keyPoints)

        # temp_img = self.original
        # # add keypoints to image, then save or show user
        # for coord, near_front in zip(keyPoints, is_foreground):
        #     x, y = coord.ravel()
        #     centre = ( int(x) + self.xmargin, int(y) + self.ymargin )

        #     # colour in BGR
        #     colour = (255, 0, 0)
        #     if near_front:
        #         colour = (0, 0, 255)

        #     cv2.circle(temp_img, centre, radius=2, color=colour, thickness=-1)

        if vid_writer is not None:
            vid_writer.write(self.original)
        else:
            cv2.imshow(title, self.original)
            cv2.waitKey(1)