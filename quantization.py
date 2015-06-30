# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from skimage import io;
from sklearn.utils import shuffle
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from time import time

@adapt_rgb(each_channel)
def equalize_each_rgb(image):
    return equalize_hist(image)


def kmeans_quantize(img, n_colors=8):

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1]
    img = np.array(img, dtype=np.float64) / 255.0

    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    print("something %d" % len(image_array))
    image_array_sample = shuffle(image_array, random_state=0)[:(len(image_array) / 4)]
    #image_array_sample = image_array
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))


    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))

    codebook = kmeans.cluster_centers_
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def fig_label(fig, image, label):
    fig.set_title(label)
    fig.axis('off')
    fig.imshow(image)


# Display all results, alongside original image
num_figs = 2

fig, (im0, im1) = plt.subplots(1, num_figs, figsize=(num_figs * 6, 6))
china = io.imread("input_samples/81060678.jpg")

fig_label(im0, china, 'original')
fig_label(im1, kmeans_quantize(china), 'quantized, kmeans')

plt.show()
