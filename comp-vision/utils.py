import matplotlib.pyplot as plt


def draw(nrows, ncols, *imgs):
    if nrows * ncols != len(imgs):
        raise RuntimeError(f'{nrows} x {ncols} != {len(imgs)}')

    plt.figure(figsize=(24, 9))
    for i, (title, img) in enumerate(imgs, start=1):
        fig = plt.subplot(nrows, ncols, i)
        if len(img.shape) > 2:
            # this is a color image (has multiple channels)
            fig.imshow(img)
        else:
            fig.imshow(img, cmap='gray')
        fig.set_title(title)

    plt.show()


def draw_notitle(nrows, ncols, *imgs):
    if nrows * ncols != len(imgs):
        raise RuntimeError(f'{nrows} x {ncols} != {len(imgs)}')

    plt.figure(figsize=(24, 9))
    for i, img in enumerate(imgs, start=1):
        fig = plt.subplot(nrows, ncols, i)
        if len(img.shape) > 2:
            # this is a color image (has multiple channels)
            fig.imshow(img)
        else:
            fig.imshow(img, cmap='gray')

    plt.show()


def draw_notitle_noaxes(nrows, ncols, *imgs):
    if nrows * ncols != len(imgs):
        raise RuntimeError(f'{nrows} x {ncols} != {len(imgs)}')

    plt.figure(figsize=(24, 9))
    for i, img in enumerate(imgs, start=1):
        fig = plt.subplot(nrows, ncols, i)
        fig.set_axis_off()
        if len(img.shape) > 2:
            # this is a color image (has multiple channels)
            fig.imshow(img)
        else:
            fig.imshow(img, cmap='gray')

    plt.show()


def main():
    from scipy.misc import imread
    import cv2

    img1 = imread('/data/comp-vision/curved-lane.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    img2 = imread('/data/comp-vision/test4.jpg')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    img3 = imread('/data/comp-vision/test6.jpg')
    img4 = imread('/data/comp-vision/signs_vehicles_xygrad.png')

    draw(
        2, 2,
        ('Curved Lane', gray1),
        ('Test 4', gray2),
        ('Test 6', img3),
        ('Signs', img4)
    )


if __name__ == '__main__':
    main()
