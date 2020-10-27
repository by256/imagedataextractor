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

image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))
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

merged_hor_lines = []
merged_ver_lines = []

d = 2

# merge all horizontal lines that do not have vertical lines between them
for i, line_i in enumerate(hor_lines):

    # find hor lines which share the same y coord
    same_hor_lines = hor_lines[((hor_lines[:, 1] - d) < line_i[1]) & (line_i[1] < (hor_lines[:, 1] + d)), :]
    common_coord = int(np.mean(same_hor_lines[:, 1]))
    if len(same_hor_lines) == 1:
        continue
    print('same_hor_lines')
    print(same_hor_lines)
    print('common_coord', common_coord)

    # find ver lines that share a y coord with hor lines
    bottom_cond = (common_coord-d <= ver_lines[:, 1]) & (ver_lines[:, 1] <= common_coord+d)
    top_cond = (common_coord-d <= ver_lines[:, 3]) & (ver_lines[:, 3] <= common_coord+d)
    valid_ver_lines = ver_lines[bottom_cond | top_cond]
    if len(valid_ver_lines) == 0:
        # if there are no such ver lines, merge all same_hor_lines
        merged_line = [np.min(same_hor_lines[:, 0]), common_coord, np.max(same_hor_lines[:, 2]), common_coord]
        print('merged_line')
        print(merged_line)
    else:
        sort_idx = np.argsort(valid_ver_lines[:, 0])
        valid_ver_lines = valid_ver_lines[sort_idx, :]
        start_ver = np.array([[0, valid_ver_lines[0, 1], 0, valid_ver_lines[0, 3]]])
        end_ver = np.array([[image.shape[1],  valid_ver_lines[-1, 1], image.shape[1], valid_ver_lines[-1, 3]]])
        valid_ver_lines = np.concatenate([start_ver, valid_ver_lines, end_ver], axis=0)
        print('valid_ver_lines')
        print(valid_ver_lines)

        for j in range(len(valid_ver_lines) - 1):
            line_a = valid_ver_lines[j, :]
            line_b = valid_ver_lines[j+1, :]
            # merge hor lines with x coords between line a and line b
            start_x, end_x = line_a[0], line_b[0]
            hor_lines_between = same_hor_lines[(same_hor_lines[:, 0] >= start_x) & (same_hor_lines[:, 0] <= end_x)]
            print(j, j+1)
            print(hor_lines_between)
            if len(hor_lines_between) == 0:
                continue
            elif len(hor_lines_between) == 1:
                # merged_hor_lines.append([hor_lines_between[0][0], int(common_coord), hor_lines_between[0][2], int(common_coord)])  # !!!!!
                # should this line be extended to the length between the two vertical lines?
                extended_line = [hor_lines_between[0][0], hor_lines_between[0][1], line_b[2], hor_lines_between[0][3]]
                merged_hor_lines.append(extended_line)
            else:
                hor_lines_between = np.array(hor_lines_between)
                # print('hor_lines_between', hor_lines_between.shape)
                merged_hor_line = [np.min(hor_lines_between[:, 0]), common_coord, np.max(hor_lines_between[:, 2]), common_coord]
                merged_hor_lines.append(merged_hor_line)


    # temp_image = image.copy()

    # # for x1, y1, x2, y2 in same_hor_lines:
    # for x1, y1, x2, y2 in merged_hor_lines:
    #     cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)
    # for x1, y1, x2, y2 in valid_ver_lines:
    #     cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)

    # plt.imshow(temp_image)
    # plt.show()
    # break

    print('\n')

merged_hor_lines = np.unique(merged_hor_lines, axis=0)

temp_image = image.copy()
for x1, y1, x2, y2 in merged_hor_lines:
    cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)
for x1, y1, x2, y2 in valid_ver_lines:
    cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)
plt.imshow(temp_image)
plt.show()

