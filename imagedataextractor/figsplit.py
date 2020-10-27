import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def area(points):
    points = np.squeeze(points)
    x, y = points[:, 0], points[:, 1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# image = cv2.imread('/home/by256/Downloads/split.jpg')
image = cv2.imread('/home/by256/Documents/Projects/imagedataextractor/examples/figsplit/10.1016.j.aca.2017.02.004.gr1.png')
# image = cv2.imread('/home/by256/Documents/Projects/imagedataextractor/examples/figsplit/10.1016.j.aca.2017.09.014.gr2.png')

h, w = image.shape[:2]
print('h, w', h, w)
max_idx = np.argmax([h, w])
print('h, w', h/image.shape[:2][max_idx], w/image.shape[:2][max_idx])
scale = 2000
new_h, new_w = int(scale*h/image.shape[:2][max_idx]), int(scale*w/image.shape[:2][max_idx])
h, w = image.shape[:2]
print('new_h, new_w', new_h, new_w)
image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))
image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# _, image_bw = cv2.threshold(image_bw, 200, 255, cv2.THRESH_BINARY)
# image_bw = cv2.bilateralFilter(image_bw, 7, 50, 50)
image_bw = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_bw)
# image_bw = cv2.bilateralFilter(image_bw, 7, 50, 50)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image, cmap='gray')
axes[1].imshow(image_bw, cmap='gray')
plt.show()

def auto_canny(x, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(x)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print(lower, upper)
    edged = cv2.Canny(x, lower, upper)
    # return the edged image
    return edged

# edges = cv2.Canny(image_bw, 50, 255)
edges = auto_canny(image_bw)
edges = cv2.dilate(edges, np.ones(shape=(7, 7), dtype=np.uint8), iterations=1)
plt.imshow(edges)
plt.show()

hor_lines = []
ver_lines = []

# get hor and ver lines
minLineLength = 1#int(np.max(image.shape) * 0.2)
maxLineGap = 100
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
# print('lines', lines.shape)
# lines = HoughBundler().process_lines(lines)
# print('lines', lines.shape)
# lines = np.reshape(lines, newshape=(lines.shape[0], 1, -1), )
# print('lines', lines.shape)
lines = np.squeeze(lines)



temp_image = image.copy()
for x1, y1, x2, y2 in lines:
    cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 3)
plt.imshow(temp_image)
plt.show()



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

d = 3

# merge all horizontal lines that do not have vertical lines between them
for i, line_i in enumerate(hor_lines):

    # find hor lines which share the same y coord
    same_hor_lines = hor_lines[((hor_lines[:, 1] - d) < line_i[1]) & (line_i[1] < (hor_lines[:, 1] + d)), :]
    common_coord = int(np.mean(same_hor_lines[:, 1]))
    if len(same_hor_lines) == 1:
        continue
    # print('same_hor_lines')
    # print(same_hor_lines)
    # print('common_coord', common_coord)

    # find ver lines that share a y coord with hor lines
    bottom_cond = (common_coord-d <= ver_lines[:, 1]) & (ver_lines[:, 1] <= common_coord+d)
    top_cond = (common_coord-d <= ver_lines[:, 3]) & (ver_lines[:, 3] <= common_coord+d)
    valid_ver_lines = ver_lines[bottom_cond | top_cond]
    if len(valid_ver_lines) == 0:
        # if there are no such ver lines, merge all same_hor_lines
        merged_line = [np.min(same_hor_lines[:, 0]), common_coord, np.max(same_hor_lines[:, 2]), common_coord]
        # print('merged_line')
        # print(merged_line)
    else:
        sort_idx = np.argsort(valid_ver_lines[:, 0])
        valid_ver_lines = valid_ver_lines[sort_idx, :]
        start_ver = np.array([[0, valid_ver_lines[0, 1], 0, valid_ver_lines[0, 3]]])
        end_ver = np.array([[image.shape[1],  valid_ver_lines[-1, 1], image.shape[1], valid_ver_lines[-1, 3]]])
        valid_ver_lines = np.concatenate([start_ver, valid_ver_lines, end_ver], axis=0)
        # print('valid_ver_lines')
        # print(valid_ver_lines)

        for j in range(len(valid_ver_lines) - 1):
            line_a = valid_ver_lines[j, :]
            line_b = valid_ver_lines[j+1, :]
            # merge hor lines with x coords between line a and line b
            start_x, end_x = line_a[0], line_b[0]
            hor_lines_between = same_hor_lines[(same_hor_lines[:, 0] >= start_x) & (same_hor_lines[:, 0] <= end_x)]
            # print(j, j+1)
            # print(hor_lines_between)
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

# merge all vertical lines that do not have horizontal lines between them
for i, line_i in enumerate(ver_lines):

    # find ver lines which share the same x coord
    same_ver_lines = ver_lines[((ver_lines[:, 0] - d) < line_i[0]) & (line_i[0] < (ver_lines[:, 0] + d)), :]
    common_coord = int(np.mean(same_ver_lines[:, 0]))
    if len(same_ver_lines) == 1:
        continue
    # print('same_ver_lines')
    # print(same_ver_lines)
    # print('common_coord', common_coord)

    # find hor lines that share a x coord with ver lines
    bottom_cond = (common_coord-d <= hor_lines[:, 0]) & (hor_lines[:, 0] <= common_coord+d)
    top_cond = (common_coord-d <= hor_lines[:, 2]) & (hor_lines[:, 2] <= common_coord+d)
    valid_hor_lines = hor_lines[bottom_cond | top_cond]
    if len(valid_hor_lines) == 0:
        # if there are no such hor lines, merge all same_ver_lines
        merged_line = [common_coord, np.min(same_ver_lines[:, 1]), common_coord, np.max(same_ver_lines[:, 3])]
        # print('merged_line')
        # print(merged_line)
    else:
        sort_idx = np.argsort(valid_hor_lines[:, 1])
        valid_hor_lines = valid_hor_lines[sort_idx, :]
        start_hor = np.array([[valid_hor_lines[0, 0], 0, valid_hor_lines[0, 2], 0]])
        end_hor = np.array([[valid_hor_lines[-1, 0], image.shape[0], valid_hor_lines[-1, 2], image.shape[0]]])
        # print('start_hor', start_hor.shape)
        # print('valid_hor_lines', valid_hor_lines.shape)
        # print('end_hor', end_hor.shape)

        valid_hor_lines = np.concatenate([start_hor, valid_hor_lines, end_hor], axis=0)
        # print('valid_hor_lines')
        # print(valid_hor_lines)

        for j in range(len(valid_hor_lines) - 1):
            line_a = valid_hor_lines[j, :]
            line_b = valid_hor_lines[j+1, :]
            # merge ver lines with y coords between line a and line b
            start_y, end_y = line_a[1], line_b[1]
            ver_lines_between = same_ver_lines[(same_ver_lines[:, 1] >= start_y) & (same_ver_lines[:, 1] <= end_y)]
            # print(j, j+1)
            # print(ver_lines_between)
            if len(ver_lines_between) == 0:
                continue
            elif len(ver_lines_between) == 1:
                # merged_ver_lines.append([common_coord, ver_lines_between[0][1], common_coord, ver_lines_between[0][3]])
                # should this line be extended to the length between the two vertical lines?
                extended_line = [ver_lines_between[0][0], ver_lines_between[0][1], ver_lines_between[0][2], line_b[3]]
                merged_ver_lines.append(extended_line)
            else:
                ver_lines_between = np.array(ver_lines_between)
                # print('ver_lines_between', ver_lines_between.shape)
                merged_ver_line = [common_coord, np.max(ver_lines_between[:, 1]), common_coord, np.min(ver_lines_between[:, 3])]
                merged_ver_lines.append(merged_ver_line)

    print('\n')

merged_hor_lines = np.unique(merged_hor_lines, axis=0)
merged_ver_lines = np.unique(merged_ver_lines, axis=0)


temp_image = image.copy()
for x1, y1, x2, y2 in merged_hor_lines:
    cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)
for x1, y1, x2, y2 in merged_ver_lines:
    cv2.line(temp_image, (x1,y1), (x2,y2), (0,255,0), 8)
plt.imshow(temp_image)
plt.show()

