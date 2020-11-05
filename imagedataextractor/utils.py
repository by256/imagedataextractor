import cv2


def get_contours(x):
    contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2:
        contours = contours[0][0]
    elif len(contours) == 3:
        contours = contours[1][0]
    return contours