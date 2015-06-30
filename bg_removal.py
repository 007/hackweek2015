import numpy as np
from scipy import ndimage
from skimage import io as img_io, img_as_float, img_as_ubyte
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.filters import sobel, sobel_v, gaussian_filter, threshold_adaptive
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.transform import rescale
from skimage.morphology import disk, opening, skeletonize, binary_closing
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import felzenszwalb, slic, quickshift, mark_boundaries, relabel_sequential, clear_border
from matplotlib import pyplot as plt

# adapt single-channel filter to multi-channel mode
@adapt_rgb(each_channel)
#def equalize_each_rgb(image):
#    p2, p98 = np.percentile(image, (2, 98))
#    return rescale_intensity(image, in_range=(p2, p98))
#    return equalize_hist(image)
def equalize_each_rgb(image):
    return equalize_adapthist(image)

@adapt_rgb(each_channel)
def sobel_each(image):
    return sobel(image)


@adapt_rgb(each_channel)
def opening_each(image, conv):
    return opening(image, conv)

@adapt_rgb(each_channel)
def skeletonize_each(image):
    return skeletonize(image)


def fig_label_segments(fig, image, segments, label):
#    labels, _ = ndimage.label(segments)
#    image_label_overlay=label2rgb(labels, image=image)
#    image_label_overlay=mark_boundaries(image, segments)
    image_label_overlay = image
    fig.set_title(label)
    fig.axis('off')
    fig.imshow(image_label_overlay)
    print ("%s number of segments: %d" % (label, len(np.unique(segments))))


def show_entropy(fig, image, scale):
    blur_size = 3  # Standard deviation in pixels.

    img1 = sobel_each(image)
    img1 = img_as_float(img1)
    img1 = img1 * scale
    img1 = img_as_ubyte(img1)
    img1 = opening_each(img1, disk(2))
    img1 = rgb2gray(img1)
    output = threshold_adaptive(img1, 3)
    output = ndimage.morphology.binary_fill_holes(output)
    output = binary_closing(output, disk(9))
    output = ndimage.morphology.binary_fill_holes(output)
#    output = skeletonize(img1)
    #img1 = img_as_float(img1)
    #blurred = gaussian_filter(image, blur_size)
    #highpass = img1 - blurred
#    output = equalize_each_rgb(img1)
#    output = ndimage.morphology.binary_fill_holes(output)
    fig_label_segments(fig, output, img1, 'sobel')


def segment_quick(fig, image):
    segmented = quickshift(image, kernel_size=3, max_dist=60, ratio=0.5)
    fig_label_segments(fig, image, segmented, 'quick')


def segment_slic(fig, image):
    segmented = slic(image, n_segments=8, compactness=2, sigma=1, enforce_connectivity=True)
    fig_label_segments(fig, image, segmented, 'slic')


def segment_fz(fig, image):
    segmented = felzenszwalb(image, scale=500, sigma=0.5, min_size=50)
    fig_label_segments(fig, image, segmented, 'felzenszwalb')


def get_input_image(fname):
    # read image without downsampling
    input_image = img_io.imread(fname)

    input_image = ndimage.median_filter(input_image, 3)

    # normalize / expand color range to maximize contrast
#    input_image = equalize_each_rgb(input_image)

    # scale image up by 2x for better visualization
    input_image = rescale(input_image, 2)

    return input_image

def show_orig(fig):
    #image = get_input_image('input_samples/27845089.jpg')
    image = get_input_image('input_samples/76969215.jpg')
    fig.set_title('original')
    fig.axis('off')
    fig.imshow(image)
    return image


def km_segmentation(image, n_segments=32, ratio=50, max_iter=100):
    # initialize on grid:
    height, width = image.shape[:2]
    # approximate grid size for desired n_segments
    step = np.sqrt(height * width / n_segments)
    grid_y, grid_x = np.mgrid[:height, :width]
    means_y = grid_y[::step, ::step]
    means_x = grid_x[::step, ::step]

    means_color = image[means_y, means_x, :]
    means = np.dstack([means_y, means_x, means_color]).reshape(-1, 5)
    image = np.dstack([grid_y, grid_x, image * ratio])

    nearest_mean = np.zeros((height, width), dtype=np.int)
    distance = np.ones((height, width), dtype=np.float) * np.inf
    for i in range(max_iter):
        print("iteration %d" % i)
        nearest_mean_old = nearest_mean.copy()
        # assign pixels to means
        for k, mean in enumerate(means):
            # compute windows:
            y_min = int(max(mean[0] - 2 * step, 0))
            y_max = int(min(mean[0] + 2 * step, height))
            x_min = int(max(mean[1] - 2 * step, 0))
            x_max = int(min(mean[1] + 2 * step, height))
            search_window = image[y_min:y_max + 1, x_min:x_max + 1]
            dist_mean = np.sum((search_window - mean) ** 2, axis=2)
            assign = distance[y_min:y_max + 1, x_min:x_max + 1] > dist_mean
            nearest_mean[y_min:y_max + 1, x_min:x_max + 1][assign] = k
            distance[y_min:y_max + 1, x_min:x_max + 1][assign] = dist_mean[assign]
        if (nearest_mean == nearest_mean_old).all():
            break
        # recompute means:
        means = [np.bincount(nearest_mean.ravel(), image[:, :, j].ravel())
                for j in range(5)]
        in_mean = np.bincount(nearest_mean.ravel())
        means = (np.vstack(means) / in_mean).T
    return nearest_mean

def segment_km(fig, image):
    segmented = km_segmentation(image)
    fig_label_segments(fig, image, segmented, 'kmeans')


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18,6))

image = show_orig(ax0)

#segment_km(ax1, image)



#segment_fz(ax1, image)
show_entropy(ax1, image, 0.2)
show_entropy(ax2, image, 0.1)
#segment_slic(ax2, image)

plt.show()
