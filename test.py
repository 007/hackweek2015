from scipy import ndimage
from skimage import data, io, filters
from skimage.color import label2rgb
from skimage.segmentation import felzenszwalb, slic
from matplotlib import pyplot as plt


def segment_slic(fig, image):
    segmented = slic(image, n_segments=4, convert2lab=True, enforce_connectivity=True)

#    segmented = ndimage.binary_fill_holes(segmented - 1)
    labels, _ = ndimage.label(segmented)
    image_label_overlay=label2rgb(labels, image=image)
    fig.set_title('slic')
    fig.axis('off')
    fig.imshow(image_label_overlay)
#    io.show()

def segment_fz(fig, image):
    segmented = felzenszwalb(image, scale=100, sigma=0.5, min_size=30)
#    segmented = felzenszwalb(image)

    segmented = ndimage.binary_fill_holes(segmented - 1)
    labels, _ = ndimage.label(segmented)
    image_label_overlay=label2rgb(labels, image=image)
    fig.set_title('felzenszwalb')
    fig.axis('off')
    fig.imshow(image_label_overlay)
#    io.imshow(image_label_overlay)
#    io.show()

input_image = data.astronaut()[::2, ::2]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))

segment_fz(ax1, input_image)
segment_slic(ax2, input_image)

#margins = dict(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
#fig.subplots_adjust(**margins)

plt.show()
