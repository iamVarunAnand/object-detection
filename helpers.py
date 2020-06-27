# import the necessary packages
import imutils


def pyramid(img, scale = 1.5, min_size = (224, 224)):
    # yield the original image
    yield img

    # keep looping over the pyramid
    while True:
        # compute the new dimensions and resize the image
        w = int(img.shape[1] / scale)
        img = imutils.resize(img, width = w)

        # if the resized image is smaller than the minimum size, stop constructing the pyramid
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield img


def sliding_window(img, step_size, window_size):
    # slide the window across the image
    for y in range(0, img.shape[0] - window_size[1], step_size):
        for x in range(0, img.shape[1] - window_size[0], step_size):
            # yield the current window
            yield (x, y, img[y: y + window_size[1], x: x + window_size[0]])
