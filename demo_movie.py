import sys
sys.path.insert(0, 'python')
from pathlib import Path
import cv2
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from collections import defaultdict


body_estimation = Body('model/body_pose_model.pth')

data_root = './images'
test_movie = 'demo_movie.mp4'
cap = cv2.VideoCapture(str(Path(data_root, test_movie)))

n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video = cv2.VideoWriter('video.avi',
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        fps,
                        (width, height))
recoder = defaultdict(list)
recoder_keys = ['right_ankle_x', 'right_ankle_y',
                'left_ankle_x', 'left_ankle_y',
                'right_groin_x', 'right_groin_y',
                'left_groin_x', 'left_groin_y']

# 動画終了まで繰り返し
i_frame = 0
cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    candidate, subset = body_estimation(frame)
    canvas = copy.deepcopy(frame)
    canvas, target_points = util.draw_bodypose(canvas, candidate, subset)
    for i, t in enumerate(target_points):
        i_x = i*2
        i_y = i*2 + 1
        recoder[recoder_keys[i_x]].append(t[0])
        recoder[recoder_keys[i_y]].append(t[1])

    video.write(canvas.astype(np.uint8))

cap.release()
video.release()
pd.DataFrame(recoder).to_csv('results/positions.csv', index=False)
