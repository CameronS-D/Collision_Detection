#!/usr/bin/env python2.7
from __future__ import division
import cv2, time, os
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from Image import Image

class CollisionDetector:

    def __init__(self):

        '''
        initialise background subtractor -> used when getting grey img
        low history value gives more accurate reults, but increases CPU cost
        varThreshold is used like a confidence level when decideing if a pixel is part of the background
        varThreshold = 16 is default, value too high means objects arent detected, value too low increase CPU cost by a lot
        '''
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=12, detectShadows=False)
        self.old_img, self.old_kp, self.cluster_info = None, None, None
        self.frame_count = 0


    def setup_vid_writer(self, img, isColor=True):

        if cv2.__version__[0] == "3":
            codec, extn = 'MPEG', "avi"
        else:
            codec, extn = 'mp4v', "mp4"

        if os.getcwd().endswith("src"):
            os.chdir("..")

        if not os.path.isdir("videos"):
            os.mkdir("videos")

        if isColor:
            img_type = "colour"
            h, w = img.original.shape[:2]
        else:
            img_type = "grey"
            h, w = img.grey.shape[:2]

        filename = os.path.join(os.getcwd(), "videos", "output_" + img_type + "." + extn)
        print("Writing output to {}".format(filename))

        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(filename, fourcc, 30, (w, h), isColor=isColor)


    def filter_cluster(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d, axis=0)
        s = d/mdev
        std_devs = np.linalg.norm(s, axis=2)
        np.nan_to_num(std_devs, copy=False)
        m *= 2 ** 0.5
        return data[std_devs < m].reshape((-1, 1, 2))


    def cluster_keypoints(self, keypoints, n_clusters=3):

        cluster_info = {
            "centroids": [],
            "dims": []
        }

        if keypoints is None or len(keypoints) < n_clusters:
            return np.array([], dtype=np.float32), cluster_info

        kp_data = np.reshape(keypoints, (-1, 2)).T

        agglo = FeatureAgglomeration(n_clusters=n_clusters)
        agglo.fit(kp_data)

        all_clusters = [ [] for _ in range(n_clusters) ]
        for kp, cluster_num in zip(keypoints, agglo.labels_):
            all_clusters[cluster_num].append(kp)

        for cluster in all_clusters[:]:
            try:
                cl = self.filter_cluster(np.array(cluster))
                xmax, ymax = cl.max(axis=0)[0]
                xmin, ymin = cl.min(axis=0)[0]
            except ValueError:
                print("Removing cluster of length ", len(cluster))
                all_clusters.remove(cluster)
                print("Tracking", len(all_clusters), "clusters")
                continue


            w = int(xmax - xmin)
            h = int(ymax - ymin)
            y_ctr = (ymin + ymax) / 2
            x_ctr = (xmin + xmax) / 2

            centroid = (int(x_ctr), int(y_ctr))

            cluster_info["centroids"].append(centroid)
            cluster_info["dims"].append((w, h))

        return np.array(keypoints, dtype=np.float32), cluster_info


    # def cluster_keypoints(self, keyPoints, dist_threshold, max_cluster_size):

    #     # sort based on distance to origin
    #     keyPoints = keyPoints.tolist()
    #     x_ctr = (self.w - self.xmargin) / 2
    #     y_ctr = (self.h - self.ymargin) / 2
    #     kp_sorted = sorted(keyPoints, key=lambda elem : elem[0][0]**2 + elem[0][1]**2)

    #     clustered_points = []
    #     current_cluster = []

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
    #                 current_cluster = []

    #     return np.array(clustered_points, dtype=np.float32)


    def get_new_keypoints(self, img, old_kp=None):
        '''
        Parameters: image
                    list of keypoints that were tracked by OF on the last iteration
        '''

        # Get new keypoints directly from grey img
        keypoints = cv2.goodFeaturesToTrack(img.grey, maxCorners=1500, qualityLevel=0.1, minDistance=10)

        if keypoints is not None and len(keypoints) > 3:
            keypoints = keypoints[::3, :, :]

        if old_kp is not None:
            try:
                keypoints = np.append(keypoints, old_kp, axis=0)
            except ValueError:
                keypoints = old_kp

        # run clustering to reduce the amount of points OF has to track and group points into potential objects
        # clustered_kp = self.cluster_keypoints(keypoints, dist_threshold=40, max_cluster_size=100)

        clustered_kp, cluster_info = self.cluster_keypoints(keypoints)

        return clustered_kp, cluster_info

    def depth_estimation(self, kp_prev, kp_current):
        ''' Parameters: Key points matched by OF in 2 consecutive frames '''

        if len(kp_prev) == 0 or len(kp_prev) != len(kp_current):
            return None

        # calculate how far each keypoint has moved over a frame
        distances = [np.linalg.norm(p1 - p0) for p0, p1 in zip(kp_prev, kp_current)]

        # return bool array to show which points moved further than given threshold
        if self.frame_count % 10 == 0:
            print("Average distance moved: {:.3f}, max moved: {:.3f}".format(np.average(distances), max(distances)))
        return distances > np.array(20, dtype=np.float32) # np.median(distances)


    def proximity_estimation(self, cluster_info, old_img, new_img, scale_threshold):

        obstacles = {
            "centroids": [],
            "dims": []
        }
        centroids = cluster_info["centroids"]
        dims = cluster_info["dims"]

        for (x, y), (width, height) in zip(centroids, dims):
            if width * height < 1000:
                continue
            w = width * 1.2/9 * 20
            h = height * 1.2/9 * 20
            x_min = max(0, int(x - w / 2))
            x_max = int(x_min + w)
            y_min = max(0, int(y - h / 2))
            y_max = int(y_min + h)

            prev_temp = old_img.grey[y_min:y_max, x_min:x_max]

            scales = np.arange(1.0, 1.6, 0.1)
            scales = [1.0, 1.3, 1.5, 1.7, 1.9]

            best_scale, best_scale_score = 0, np.Inf

            for scale in scales:
                needle_template = cv2.resize(prev_temp, dsize=(0, 0), fx=scale, fy=scale)

                new_h, new_w = needle_template.shape[:2]
                x_min = max(0, int(x - new_w / 2))
                x_max = int(x + new_w / 2)
                y_min = max(0, int(y - new_h / 2))
                y_max = int(y + new_h / 2)

                haystack_template = new_img.grey[y_min:y_max, x_min:x_max]
                haystack_template = cv2.resize(haystack_template, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
                # result = cv2.matchTemplate(needle_template, haystack_template, method=cv2.TM_SQDIFF_NORMED )
                # score, _, _, _ = cv2.minMaxLoc(result)
                # score *= scale ** (-2)
                score = ((haystack_template - needle_template) ** 2).mean(axis=None)

                if scale == 1.0:
                    control_score = score

                if score < best_scale_score:
                    best_scale_score = score
                    best_scale = scale

            if best_scale > scale_threshold and best_scale_score < 0.8 * control_score:
                obstacles["centroids"].append((x, y))
                obstacles["dims"].append((width, height))

        return obstacles

    def check_for_empty_space(self, arr, w, h):

        sub_w = w // 2
        sub_h = h // 2

        points = []

        if sub_w < 50 or sub_h < 50:
            return points

        img_h, img_w = self.old_img.grey.shape[:2]

        for i in range(img_h // sub_h):
            for j in range(img_w // sub_w):
                sub_sec = arr[i*sub_h:(i+1)*sub_h, j*sub_w:(j+1)*sub_w]

                if 1 not in sub_sec:
                    points.append(np.array([i*sub_h + sub_h // 2, j*sub_w + sub_w // 2]))

        if len(points) == 0:
            points += self.check_for_empty_space(sub_sec, sub_w, sub_h)

        return points


    def get_safe_point(self, obstacles):

        if len(obstacles["centroids"]) == 0:
            return None

        img_h, img_w = self.old_img.grey.shape[:2]

        if self.frame_count % 30 == 0:
            # Only reset heatmap every few frames -> this way objects that are only captured briefly are remembered
            self.heatmap = np.zeros(shape=(img_h, img_w))

        centroids = obstacles["centroids"]
        dims = obstacles["dims"]

        for (x, y), (w, h) in zip(centroids, dims):
            w *= 1.2
            h *= 1.2
            x_min = max(0, int(x - w / 2))
            x_max = int(x_min + w)
            y_min = max(0, int(y - h / 2))
            y_max = int(y_min + h)

            self.heatmap[y_min:y_max, x_min:x_max] = 1

        safepoints = np.array(self.check_for_empty_space(self.heatmap, img_w, img_h))

        if len(safepoints) > 0:
            ctr_x = (img_w-1) // 2
            ctr_y = (img_h-1) // 2
            dists = np.sum((safepoints - np.array([ctr_y, ctr_x])) ** 2, axis=1   )

            closest_pnt_idx = np.argmin(dists)
            y, x = safepoints[closest_pnt_idx]
            return (x, y)
        return None


    def process_frame(self, frame):

        t0 = time.time()

        new_img = Image(frame, self.bgs)

        if self.old_img is None:
            self.vid_writer_color = self.setup_vid_writer(new_img, isColor=True)
            self.vid_writer_grey = self.setup_vid_writer(new_img, isColor=False)
            self.old_img = new_img
            self.old_kp, self.cluster_info = self.get_new_keypoints(new_img)
            img_h, img_w = self.old_img.grey.shape[:2]
            self.heatmap = np.zeros(shape=(img_h, img_w))
            self.frame_count += 1
            return

        old_img, old_kp, cluster_info = self.old_img, self.old_kp, self.cluster_info

        if self.frame_count % 30 == 0:
                old_kp, cluster_info = self.get_new_keypoints(new_img, old_kp=old_kp)
        else:
            old_kp, cluster_info = self.cluster_keypoints(old_kp)

        if len(old_kp) != 0:

            new_kp, status, err = cv2.calcOpticalFlowPyrLK(old_img.grey, new_img.grey, old_kp, None, maxLevel=3)

            # select points that were matched by OF in new frame
            matched_new_kp = new_kp[status==1]
            matched_old_kp = old_kp[status==1]
            # reshape for use with other functions
            matched_new_kp = matched_new_kp.reshape((-1, 1, 2))
            matched_old_kp = matched_old_kp.reshape((-1, 1, 2))

            # get bool array stating which points are estimated to be in the foreground
            fg = self.depth_estimation(matched_old_kp, matched_new_kp)
            obstacles = self.proximity_estimation(cluster_info, old_img, new_img, scale_threshold=1.2)
            old_kp = matched_new_kp
            safe_pnt = self.get_safe_point(obstacles)

        else:
            old_kp, cluster_info = self.get_new_keypoints(new_img)
            obstacles = None
            fg = None
            safe_pnt = None

        new_img.add_features(obstacles, old_kp, fg, safe_pnt)
        # new_img.show(vid_writer=self.vid_writer_color, isColor=True)
        # new_img.show(vid_writer=self.vid_writer_grey, isColor=False)
        new_img.show()

        if self.frame_count % 10 == 0:
            time_taken = time.time() - t0
            print("Took {:.3f} seconds to process frame {}".format(time_taken, self.frame_count))

        self.old_img = new_img
        self.old_kp, self.cluster_info = old_kp, cluster_info
        self.frame_count += 1


if __name__ == "__main__":
    CD = CollisionDetector()
    # vidstream = cv2.VideoCapture("../videos/beach_with_trees.mp4")
    vidstream = cv2.VideoCapture(0)

    if not vidstream.isOpened():
        raise Exception

    while True:
        success, img = vidstream.read()
        if not success:
            print("\nVideo complete. Output written to output.mp4 or output.avi")
            break
        CD.process_frame(img)
