import sys
import os.path as op
import argparse
import logging

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
this_file = op.abspath(this_file)

if __name__ == '__main__':
    # When run as script, modify path assuming absolute import
    sys.path.append(op.join(op.dirname(this_file), '..'))

from mmod.utils import init_logging
from mmod.tax_utils import create_collage, create_inverted
from mmod.imdb import ImageDatabase


def main():
    init_logging()
    parser = argparse.ArgumentParser(description='Create collage (and inverted) for an image database.')
    parser.add_argument('dbpath', metavar='DB_PATH', help='Path to an image db')
    parser.add_argument('--seed', help='Random seed', type=int)
    parser.add_argument('--max_label', help='Maximum number of samples per-label', type=int, default=100)
    parser.add_argument('--max_children', help='Maximum number of samples per children label', type=int, default=200)
    parser.add_argument('--target_size', help='Target size of thumbnails to rescale', type=int, default=100)
    parser.add_argument('--start_label', help='First label to create collage')
    parser.add_argument('--end_label', help='Last label to create collage')

    args = parser.parse_args()
    args = vars(args)
    imdb = ImageDatabase(args['dbpath'])
    assert imdb.is_prototxt or imdb.is_tsv, "{} is not supported for collage".format(imdb)

    start_label = args['start_label']
    assert not start_label or start_label in imdb.cmap, "{} is not in {}".format(start_label, imdb)
    end_label = args['end_label']
    assert not end_label or end_label in imdb.cmap, "{} is not in {}".format(end_label, imdb)
    max_children = args['max_children']
    max_label = args['max_label']

    if not imdb.is_inverted:
        labelmap = imdb.cmapfile
        inverted_file = imdb.inverted_path
        shuffle_file = imdb.shuffle_path
        assert not op.isfile(inverted_file), "inverted file already exists: {} db: {}".format(inverted_file, imdb)
        logging.info("Create missing inverted file: {} and reloading db: {}".format(inverted_file, imdb))
        create_inverted(imdb, path=inverted_file, shuffle=shuffle_file, labelmap=labelmap,
                        create_shuffle=not op.isfile(shuffle_file), create_labelmap=not op.isfile(labelmap))
        if not max_children and not max_label:
            logging.info("No collage requested")
            return imdb
        imdb = ImageDatabase(args['dbpath'])

    assert imdb.is_inverted, "Could not create inverted index, iteration be too slow without it"

    if not max_children and not max_label:
        logging.info("No collage requested")
        return imdb

    logging.info("Create collage for {}".format(imdb))
    create_collage(
        imdb,
        seed=args['seed'],
        max_children=max_children,
        max_label=max_label,
        target_size=args['target_size'],
        start_label=start_label,
        end_label=end_label,
    )

    return imdb


if __name__ == '__main__':
    _imdb = main()
