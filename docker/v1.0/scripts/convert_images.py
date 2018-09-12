import os
# noinspection PyPackageRequirements
from PIL import Image
import numpy as np
import argparse
try:
    import caffe
except ImportError:
    caffe = None
try:
    import h5py
except ImportError:
    h5py = None


def create_imagedata(path, imagedata, imagemean, counts, max_keep_per_label=np.inf):
    """Create imagedata caffe structure and their mean
    Each image is labeled by its directory name
    """
    label = None
    localroot = None
    for s0 in os.listdir(path):
        s0 = os.path.join(path, s0)
        if os.path.isdir(s0):
            create_imagedata(s0, imagedata, imagemean, counts)
            continue
        _, file_extension = os.path.splitext(s0)
        if file_extension.lower() not in [".jpg", ".png"]:
            continue

        if label is None or localroot is None:
            # label comes from immediate parent
            parentdir = os.path.dirname(s0)
            label = os.path.basename(os.path.normpath(parentdir))
            # store source and mean file in the grandparent
            localroot = os.path.dirname(parentdir)

        if label not in counts:
            counts[label] = 1
        elif counts[label] >= max_keep_per_label:
            continue
        else:
            counts[label] += 1

        if localroot not in imagemean:
            imagedata[localroot] = [(s0, label)]
            imagemean[localroot] = [np.int32(Image.open(s0)), 1]
        else:
            curmean = imagemean[localroot]
            curmean[0] += np.int32(Image.open(s0))
            curmean[1] += 1
            imagedata[localroot].append((s0, label))


def file_name(subpath, path, base_name):
    """Return the base name for a subdirectory with respect to the root path
    """
    rel = os.path.relpath(subpath, path)
    if rel == ".":
        return os.path.join(path, base_name)
    base = rel.replace("\\", "/").replace("/", "_").replace(".", "_")
    return os.path.join(path, base + "_" + base_name)


def main():
    parser = argparse.ArgumentParser(description='Process images in a path recursively and prepare Caffe data.')

    parser.add_argument('-keep', '--keep', help='Maximum number of images to keep for each label',
                        type=float, default=np.inf)
    parser.add_argument('root_path', metavar='PATH', help='path to the images dataset')

    args = parser.parse_args()
    root_path = args.root_path
    max_keep_per_label = args.keep

    images = {}
    means = {}
    counts = {}
    create_imagedata(root_path, images, means, counts, max_keep_per_label=max_keep_per_label)
    orig_shape = None

    for k, v in means.items():
        mean_npy = v[0] / v[1]
        if orig_shape is None:
            orig_shape = mean_npy.shape
        if len(mean_npy.shape) != 3:
            # grayscale
            mean_npy = mean_npy.reshape((1, 1, *mean_npy.shape))
        np.save(file_name(k, root_path, 'mean.npy'), mean_npy)

        if caffe:
            blob = caffe.io.array_to_blobproto(mean_npy)
            with open(file_name(k, root_path, 'mean.binaryproto'), 'wb') as f:
                f.write(blob.SerializeToString())

    for k, vs in images.items():
        with open(file_name(k, root_path, 'images.txt'), "w") as f:
            for v in vs:
                relpath = os.path.relpath(v[0], root_path).replace("\\", "/")
                f.write("{} {}\n".format(relpath, v[1]))

    if h5py:
        for k, vs in images.items():
            with open(file_name(k, root_path, 'h5_list.txt'), "w") as list_f:
                h5_path = file_name(k, root_path, 'images.h5')
                list_f.write(os.path.basename(h5_path))  # just list the h5 files here
                # HDF5 data does not support trasformation, so we should subtract the mean ourselves
                mean_npy = means[k][0] / means[k][1]
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset('data', (len(vs), 1, *orig_shape), dtype='f4')
                    f.create_dataset('label', (len(vs), 1), dtype='f4')
                    for idx, v in enumerate(vs):
                        img = np.float32(Image.open(v[0]))
                        f['data'][idx] = img - mean_npy  # subtract the mean
                        f['label'][idx] = float(v[1])

    return images, means

if __name__ == '__main__':
    main_results = main()
