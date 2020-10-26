import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def area(points):
    points = np.squeeze(points)
    x, y = points[:, 0], points[:, 1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

image = cv2.imread('/home/by256/Downloads/split.jpg')

image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# _, image_bw = cv2.threshold(image_bw, 200, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(image_bw, 200, 255)

hor_lines = []
ver_lines = []

# get hor and ver lines
minLineLength = int(np.max(image.shape) * 0.2)
maxLineGap = 100
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
lines = np.squeeze(lines)
for x1, y1, x2, y2 in lines:
    # gradient = (y2 - y1) / (x2 - x1)
    if y1 == y2:
        hor_lines.append((x1, y1, x2, y2))
    elif x1 == x2:
        ver_lines.append((x1, y1, x2, y2))

    # cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)

hor_lines = np.array(hor_lines)
ver_lines = np.array(ver_lines)

print(hor_lines.shape, ver_lines.shape)

d = 2

# merge all horizontal lines that do not have vertical lines between them
for i, line_i in enumerate(hor_lines):

    # find horizontal lines which share the same horizontal coords
    same_hor_lines = hor_lines[((hor_lines[:, 1] - d) < line_i[1]) & (line_i[1] < (hor_lines[:, 1] + d)), :]
    common_coord = np.mean(same_hor_lines[:, 1])
    if len(same_hor_lines) == 1:
        continue
    print('same_hor_lines')
    print(same_hor_lines)
    print('common_coord', common_coord)

    # find vertical lines that have the same common horizontal coord
    bottom_cond = (common_coord-d <= ver_lines[:, 1]) & (ver_lines[:, 1] <= common_coord+d)
    top_cond = (common_coord-d <= ver_lines[:, 3]) & (ver_lines[:, 3] <= common_coord+d)
    valid_ver_lines = ver_lines[bottom_cond | top_cond]
    if len(valid_ver_lines) == 0:
        # merge all same_hor_lines
        merged_line = [np.min(same_hor_lines[:, 0]), int(common_coord), np.max(same_hor_lines[:, 2]), int(common_coord)]
        print('merged_line')
        print(merged_line)
    else:
        sort_idx = np.argsort(valid_ver_lines[:, 0])
        valid_ver_lines = valid_ver_lines[sort_idx, :]
        start_ver = np.array([[0, 0, 0, image.shape[0]]])
        end_ver = np.array([[image.shape[1],  image.shape[0], image.shape[1], 0]])
        valid_ver_lines = np.concatenate([start_ver, valid_ver_lines, end_ver], axis=0)
        print('valid_ver_lines')
        print(valid_ver_lines)

    temp_image = image.copy()

    for x1, y1, x2, y2 in same_hor_lines:
        cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 4)
    for x1, y1, x2, y2 in valid_ver_lines:
        cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 4)

    plt.imshow(temp_image)
    plt.show()
    # break

    print('\n')


plt.imshow(image)
plt.show()