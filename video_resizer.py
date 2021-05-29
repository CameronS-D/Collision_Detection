import cv2

vidcap = cv2.VideoCapture("videos/under_bridge.mp4")

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vidwrite = cv2.VideoWriter("videos/under_bridge_resized.mp4", fourcc, 30, (1280, 720), isColor=True)

while True:
    ret, img = vidcap.read()
    if not ret:
        break

    new_img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    vidwrite.write(new_img)
