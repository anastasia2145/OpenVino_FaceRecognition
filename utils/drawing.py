import numpy as np
import cv2 as cv

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_GREEN = (0.0, 200.0, 0.0)
OUTSIDE_CONTOUR = -1

colours = np.random.randint(0, 255, 10001)


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=2):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv.circle(img, p, thickness, color, -1)
    else:
        if len(pts) > 0:
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    cv.line(img, s, e, color, thickness)
                i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted'):
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


def draw_text_on_image(img, text, bottom_left):
    intFontFace = cv.FONT_HERSHEY_SIMPLEX
    dblFontScale = 1
    intFontThickness = 2
    cv.putText(img, str(text), bottom_left, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness)