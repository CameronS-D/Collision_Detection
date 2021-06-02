#!/usr/bin/env python2.7
import cv2
import numpy as np
import time
from sklearn.cluster import FeatureAgglomeration
from Image import Image

class CollisionDetector:

    def __init__(self, filepath):

        self.vidstream = cv2.VideoCapture(filepath)
        # Setup output video writer
        if cv2.__version__[0] == "3":
            codec, extn = 'MPEG', "avi"
        else:
            codec, extn = 'mp4v', "mp4"

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.vid_writer = cv2.VideoWriter("output." + extn, fourcc, 30, (int(0.5*1280), int(0.5*720)), isColor=True)
        '''
        initialise background subtractor -> used when getting grey img
        low history value gives more accurate reults, but increases CPU cost
        varThreshold is used like a confidence level when decideing if a pixel is part of the background
        varThreshold = 16 is default, value too high means objects arent detected, value too low increase CPU cost by a lot
        '''
        self.bgs = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=12, detectShadows=False)

        self.best_matching_scales = {}

    def filter_cluster(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d, axis=0)
        s = d/mdev
        std_devs = np.linalg.norm(s, axis=2)
        np.nan_to_num(std_devs, copy=False)
        m *= 2 ** 0.5
        return data[std_devs < m].reshape((-1, 1, 2))


    def cluster_keypoints(self, keypoints, n_clusters=8):

        cluster_info = {
            "centroids": [],
            "dims": []
        }

        if keypoints is None or len(keypoints) < 3 * n_clusters:
            return np.array([], dtype=np.float32), cluster_info

        keypoints = keypoints[::3, :, :]

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
        keypoints = cv2.goodFeaturesToTrack(img.grey, maxCorners=3000, qualityLevel=0.1, minDistance=5)

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
        return distances > np.array(50, dtype=np.float32) # np.median(distances)


    def proximity_estimation(self, cluster_info, old_img, new_img, scale_threshold):

        obstacles = {
            "centroids": [],
            "dims": []
        }
        centroids = cluster_info["centroids"]
        dims = cluster_info["dims"]

        for (x, y), (w, h) in zip(centroids, dims):
            if w * h < 2500:
                continue
            x_min = max(0, x - w // 2)
            x_max = x_min + w
            y_min = max(0, y - h // 2)
            y_max = y_min + h

            prev_temp = old_img.grey[y_min:y_max, x_min:x_max]
            current_temp = new_img.grey[y_min:y_max, x_min:x_max]

            scales = np.arange(2.5, 3.5, 0.1)
            try:
                scaled_templates = [cv2.resize(prev_temp, dsize=(0, 0), fx=sf, fy=sf) for sf in scales]
            except cv2.error as e:
                continue

            best_scale, best_scale_score = 0, np.Inf

            for idx in range(len(scales)):
                template = scaled_templates[idx]
                result = cv2.matchTemplate(template, current_temp, method=cv2.TM_SQDIFF )
                score, _, _, _ = cv2.minMaxLoc(result)

                if score < best_scale_score:
                    best_scale_score = score
                    best_scale = scales[idx]

            self.best_matching_scales.setdefault(best_scale, 0)
            self.best_matching_scales[best_scale] += 1

            if best_scale > scale_threshold:
                obstacles["centroids"].append((x, y))
                obstacles["dims"].append((w, h))

        return obstacles


    def run(self):
        count = 0
        t0 = time.time()

        success, img = self.vidstream.read()

        old_img = Image(img, self.bgs)
        # get first keypoints
        old_kp, cluster_info = self.get_new_keypoints(old_img)

        while(True):
            success, img = self.vidstream.read()
            if not success:
                print("\nVideo complete. Output written to output.mp4 or output.avi")
                break
            new_img = Image(img, self.bgs)

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
                obstacles = self.proximity_estimation(cluster_info, old_img, new_img, scale_threshold=3.3)
                old_kp, cluster_info = self.get_new_keypoints(new_img, old_kp=matched_new_kp)

            else:
                old_kp, cluster_info = self.get_new_keypoints(new_img)
                obstacles = None
                fg = None

            new_img.add_features(obstacles, old_kp, fg)
            new_img.show(vid_writer=self.vid_writer)

            old_img = new_img
            # old_kp = matched_new_kp

            if count % 30 == 0:
                '''
                Periodically search for new features of interest,
                otherwise only points that were seen at the start will ever be detected
                '''
                # old_kp, cluster_info = self.get_new_keypoints(new_img, old_kp=matched_new_kp)
                # for testing purposes
                t1 = time.time()
                time_taken = t1 - t0
                vid_time = count / 30
                error = time_taken - vid_time
                print("Frame: {} Actual time: {:.1f} Vid time: {} Lag: {:.1f} secs".format(count, time_taken, vid_time, error))

            count += 1


if __name__ == "__main__":
    CD = CollisionDetector("../videos/beach_with_trees.mp4")
    CD.run()
