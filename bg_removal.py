from glob import iglob
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.colors import hex2color
from scipy import ndimage
from skimage import io as img_io, img_as_float, img_as_ubyte
from skimage.color import label2rgb, rgb2gray
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel, threshold_adaptive
from skimage.morphology import disk, opening, binary_closing, binary_opening
from skimage.transform import rescale

# adapt single-channel filter to multi-channel mode
@adapt_rgb(each_channel)
def equalize_each_rgb(image):
    return equalize_adapthist(image)


@adapt_rgb(each_channel)
def sobel_each(image):
    return sobel(image)


@adapt_rgb(each_channel)
def opening_each(image, conv):
    return opening(image, conv)


def fig_label_segments(fig, image, segments, label):
    print ("%s number of segments: %d" % (label, len(np.unique(segments))))
    image_label_overlay=label2rgb(segments, image=image, colors=(hex2color('#000000'), hex2color('#ff0000')))
    fig.set_title(label)
    fig.axis('off')
    fig.imshow(image_label_overlay)


def show_entropy(fig, image, scale):
    img1 = img_as_float(image)
    img1 = sobel_each(img1)
    img1 = img1 * scale
    img1 = img_as_ubyte(img1)
    img1 = opening_each(img1, disk(2))
    img1 = rgb2gray(img1)
    output = threshold_adaptive(img1, 3)
    output = ndimage.morphology.binary_fill_holes(output)
    output = binary_closing(output, disk(9))
    output = ndimage.morphology.binary_fill_holes(output)
    output = binary_opening(output, disk(3))
    fig_label_segments(fig, image, output, 'sobel s=' + str(scale))


def get_input_image(fname):
    # read image without downsampling
    input_image = img_io.imread(fname)
    input_image = ndimage.median_filter(input_image, 3)
    # scale image up by 2x for better visualization
    input_image = rescale(input_image, 2)
    return input_image


def show_orig(fig, fn):
    #image = get_input_image('input_samples/27845089.jpg')
    image = get_input_image(fn)
    fig.set_title('original')
    fig.axis('off')
    fig.imshow(image)
    return image


# __main__
fig_count = 3
for fn in iglob('input_samples/*.jpg'):

    fig, (ax0, ax1, ax2) = plt.subplots(1, fig_count, figsize=(fig_count * 6, 6))

    image = show_orig(ax0, fn)

    show_entropy(ax1, image, 0.15)
    show_entropy(ax2, image, 0.125)

    nr = re.compile('input_samples/(\d+).jpg')
    outname = nr.sub(r'output_samples/\1.png', fn)
    print("Saving %s to filename %s" % (fn, outname))
    plt.savefig(outname)
    plt.close()

