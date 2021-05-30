import cv2
import numpy as np
import time
from itertools import cycle
from sklearn.cluster import FeatureAgglomeration

'''
Version of CollisionDetector that has been edited to compare selected
keypoints before and after clustering
'''

class CollisionDetector:

    def __init__(self, filepath):

        self.vidstream = cv2.VideoCapture(filepath)
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.vidwriter = cv2.VideoWriter("output.mp4", fourcc, 30, (1280, 720), isColor=True)

        '''
        initialise background subtractor -> used when getting grey img
        low history value gives more accurate reults, but increases CPU cost
        varThreshold is used like a confidence level when decideing if a pixel is part of the background
        varThreshold = 16 is default, value too high means objects arent detected, value too low increase CPU cost by a lot
        '''
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=12, detectShadows=False)

        self.vidstream.read()
        success, self.old_img = self.vidstream.read()

        # set margin so only objects near the centre of the screen are tracked/detected -> cuts computation time by a lot
        h, w = self.old_img.shape[:2]
        self.h, self.w = h, w
        self.xmargin = int( 0.2 * w )
        self.ymargin = int( 0.2 * h )

        self.min_contour_area = 0.01 * h

        # get first keypoints
        self.old_grey = self.get_grey_contour_img(self.old_img)
        self.old_kp, _ = self.get_new_keypoints(self.old_grey)
        self.max_dist_moved = 0


    def get_grey_contour_img(self, image):

        # get region of interest of image
        image = image[self.ymargin:-self.ymargin, self.xmargin:-self.xmargin].copy()

        # get foreground mask from image and find contours in it
        fgMask = self.bgs.apply(image, learningRate=0.1)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # eliminate small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]

        tempImg = np.zeros((*image.shape[:2], 3), np.uint8)
        cv2.drawContours(tempImg, contours, -1, (128,255,255), -1)

        return cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)


    def cluster_keypoints(self, keypoints, n_clusters=8):

        keypoints = keypoints[::3, :, :]

        if len(keypoints) < n_clusters:
            return np.array(keypoints, dtype=np.float32), [np.array(keypoints, dtype=np.float32)]

        kp_data = np.reshape(keypoints, (-1, 2)).T

        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        agglo.fit(kp_data)

        all_clusters = [ [] for _ in range(n_clusters) ]
        for kp, cluster_num in zip(keypoints, agglo.labels_):
            all_clusters[cluster_num].append(kp)

        # for idx, cluster in enumerate(all_clusters):
        #     cluster = np.array(cluster, dtype=np.float32)
        #     x_ctr, y_ctr = np.median(cluster, axis=0)[0]

        return np.array(keypoints, dtype=np.float32), all_clusters


    # def cluster_keypoints(self, keyPoints, dist_threshold, max_cluster_size):

    #     # sort based on distance to origin
    #     keyPoints = keyPoints.tolist()
    #     x_ctr = self.w / 2 - self.xmargin
    #     y_ctr = self.h / 2 - self.ymargin
    #     kp_sorted = sorted(keyPoints, key=lambda elem : (elem[0][0])**2 + (elem[0][1])**2)

    #     clustered_points = []
    #     current_cluster = []
    #     all_clusters = []

    #     '''
    #     Here we loop through all points. If one is close enough to the next, add it to a cluster
    #     This should discard lone keypoints
    #     '''
    #     for idx in range(len(kp_sorted) - 1):
    #         x1, y1 = kp_sorted[idx][0]
    #         x2, y2 = kp_sorted[idx+1][0]

    #         dist = ( (x2 - x1)**2 + (y2 - y1)**2 ) ** 0.5

    #         if dist < dist_threshold:
    #             current_cluster.append( kp_sorted[idx] )
    #         else:
    #             if len(current_cluster) > max_cluster_size:
    #                 clustered_points += current_cluster
    #                 all_clusters.append(current_cluster)
    #                 current_cluster = []


    def get_new_keypoints(self, img_grey, old_kp=None):
        '''
        Parameters: greyscale contour image
                    list of keypoints that were tracked by OF on the last iteration
        '''

        # Get new keypoints directly from grey img
        keypoints = cv2.goodFeaturesToTrack(img_grey, maxCorners=3000, qualityLevel=0.1, minDistance=5)

        if old_kp is not None:
            try:
                keypoints = np.append(keypoints, old_kp, axis=0)
            except ValueError:
                keypoints = old_kp

        if keypoints is None:
            return [], None

        self.show_image(self.old_img, keypoints, title="Without clustering")
        # run clustering to reduce the amount of points OF has to track
        # clustered_kp, clusters = self.cluster_keypoints(keypoints, dist_threshold=40, max_cluster_size=100)
        clustered_kp, clusters = self.cluster_keypoints(keypoints)

        if len(clustered_kp) == 0:
            return keypoints, None
        return clustered_kp, clusters

    def depth_estimation(self, kp_prev, kp_current):
        ''' Parameters: Key points matched by OF in 2 consecutive frames '''

        if len(kp_prev) == 0 or len(kp_prev) != len(kp_current):
            return None

        # calculate how far each keypoint has moved over a frame
        distances = [np.linalg.norm(p1 - p0) for p0, p1 in zip(kp_prev, kp_current)]
        # if max(distances) > self.max_dist_moved:
        #     self.max_dist_moved = max(distances)
        #     print("new max dist moved = ", max(distances))

        # return bool array to show which points moved further than given threshold
        return distances > np.array(50, dtype=np.float32) # np.median(distances)

    def proximity_estimation(self):
        pass
        # TODO: Implement func based on pseudo-code in algorithm 3 in
        # https://staff.fnwi.uva.nl/a.visser/education/masterProjects/Obstacle%20Avoidance%20using%20Monocular%20Vision%20on%20Micro%20Aerial%20Vehicles_final.pdf


    def show_image(self, img, keyPoints, foreground=None, title="Image", save_vid=False):

        if foreground is None:
            foreground = [False] * len(keyPoints)

        temp_img = img.copy()
        # add keypoints to image, then save or show user
        for coord, near_front in zip(keyPoints, foreground):
            x, y = coord.ravel()
            centre = (int(x)+self.xmargin, int(y)+self.ymargin)
            # colour in BGR
            colour = (255, 0, 0)
            if near_front:
                colour = (0, 0, 255)

            cv2.circle(temp_img, centre, 1, colour, 2)

        if save_vid:
            self.vidwriter.write(temp_img)
        else:
            cv2.imshow(title, temp_img)
            cv2.waitKey(1)


    def run(self):
        count = 0
        t0 = time.time()

        while(True):
            success, img = self.vidstream.read()
            if not success:
                print("Video complete")
                break

            img_grey = self.get_grey_contour_img(img)

            if len(self.old_kp) != 0:

                new_kp, status, err = cv2.calcOpticalFlowPyrLK(self.old_grey, img_grey, self.old_kp, None, maxLevel=3)

                matched_new_kp = new_kp[status==1]
                matched_old_kp = self.old_kp[status==1]

                matched_new_kp = matched_new_kp.reshape((-1, 1, 2))
                matched_old_kp = matched_old_kp.reshape((-1, 1, 2))

                fg = self.depth_estimation(matched_old_kp, matched_new_kp)

            else:
                matched_new_kp, _ = self.get_new_keypoints(img_grey.copy())
                fg = None

            # self.show_image(img, matched_new_kp, fg, save_vid=True)
            self.old_img = img
            self.old_grey = img_grey.copy()
            self.old_kp = matched_new_kp

            if count % 15 == 0:
                self.old_kp, clusters = self.get_new_keypoints(self.old_grey, old_kp=matched_new_kp)

                # The following code is used to test/evaluate the clustering algorithm
                colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
                colourcycler = cycle(colours)

                for cluster in clusters:
                    colour = next(colourcycler)
                    for coord in cluster:

                        x, y = coord[0]
                        centre = (int(x)+self.xmargin, int(y)+self.ymargin)
                        cv2.circle(img, centre, 1, colour, 2)
                cv2.imshow("With clustering", img)
                cv2.waitKey(0)

                t1 = time.time()
                time_taken = t1 - t0
                vid_time = count / 30
                error = time_taken - vid_time
                print(f'Frame: {count} Actual time: {time_taken:.1f} Vid time: {vid_time} Lag: {error:.1f} secs')

            count += 1


if __name__ == "__main__":
    CD = CollisionDetector("videos/beach_with_trees.mp4")
    CD.run()
