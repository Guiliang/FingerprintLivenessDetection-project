import numpy as np
import os
from glob import glob
from scipy import misc

IMG_TRANSLATION_RATIO = 0.8
dirs = {
    '../data-livdet-2015/Testing/Digital_Persona/Fake': '../data-livdet-2015/Testing_augmented/Digital_Persona/Fake',
    '../data-livdet-2015/Testing/Digital_Persona/Live': '../data-livdet-2015/Testing_augmented/Digital_Persona/Live',
    '../data-livdet-2015/Training/Digital_Persona/Fake': '../data-livdet-2015/Training_augmented/Digital_Persona/Fake',
    '../data-livdet-2015/Training/Digital_Persona/Live': '../data-livdet-2015/Training_augmented/Digital_Persona/Live',
}


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

for source_dir in dirs:
    target_dir = dirs[source_dir]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = sorted(glob(os.path.join(source_dir, '*')))
    for i, f in enumerate(files):
        arr = misc.imread(f)

        w = arr.shape[1]
        h = arr.shape[0]
        dw = int(w * (1 - IMG_TRANSLATION_RATIO) / 2)
        dh = int(h * (1 - IMG_TRANSLATION_RATIO) / 2)
        new_w = w - 2 * dw
        new_h = h - 2 * dh

        patches = [
            [0, new_h, 0, new_w],       # top left
            [0, new_h, 2 * dw, w],      # top right
            [dh, h - dh, dw, w - dw],   # center
            [2 * dh, h, 0, new_w],      # bottom left
            [2 * dh, h, 2 * dw, w],     # bottom right
        ]
        k = 1
        for do_horizontal_flip in [False, True]:
            if do_horizontal_flip:
                arr = flip_axis(arr, 1)
            for h1, h2, w1, w2 in patches:
                patch = arr[h1:h2, w1:w2]

                # file.png => file_k.png
                new_f = '_{:d}.'.format(k).join(os.path.basename(f).rsplit('.', 1))
                misc.imsave(os.path.join(target_dir, new_f), patch)
                k += 1
