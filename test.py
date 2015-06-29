from scipy import ndimage
from skimage import io as img_io
from skimage.exposure import equalize_hist
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic
from matplotlib import pyplot as plt


def fig_label_segments(fig, image, segments, label):
    labels, _ = ndimage.label(segments)
    image_label_overlay=label2rgb(labels, image=image)
    fig.set_title(label)
    fig.axis('off')
    fig.imshow(image_label_overlay)

def segment_slic(fig, image):
    segmented = slic(image, n_segments=6, convert2lab=True, enforce_connectivity=True)

    segmented = ndimage.binary_fill_holes(segmented - 1)
    fig_label_segments(fig, image, segmented, 'slic')

def segment_fz(fig, image):
    segmented = felzenszwalb(image)
    segmented = felzenszwalb(image, scale=50, sigma=0.5, min_size=30)

    segmented = ndimage.binary_fill_holes(segmented - 1)
    fig_label_segments(fig, image, segmented, 'felzenszwalb')

def get_input_image(fname):

    # read image and downsample by 2x
    #input_image = img_io.imread('input_samples/27845089.jpg')[::2, ::2]

    # don't downsample
    input_image = img_io.imread(fname)

    # select medians to filter outlier
    input_image = ndimage.median_filter(input_image, 3)

    # normalize / expand color range to maximize contrast
    input_image = equalize_hist(input_image)


    # scale image up by 2x for better visualization
    input_image = rescale(input_image, 2)

#    img_io.imshow(input_image)
#    img_io.show()

    return input_image

image = get_input_image('input_samples/27845089.jpg')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

segment_fz(ax1, image)
segment_slic(ax2, image)

plt.show()
