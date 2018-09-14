import logging
import os.path as op
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import pandas as pd

from mmod.simple_parser import load_labelmap_list
from mmod.im_utils import int_rect

STANDARD_COLORS = [
    'green', 'lightgreen',
    'red', 'darksalmon',
    'darkorange', 'navajowhite',
    'deeppink', 'lightpink',
    'm', 'fuchsia',
    'chartreuse', 'lawngreen',
    'gold', 'khaki',
    'crimson', 'palevioletred',
]


_VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.png']


def draw_rect(ax, all_cls, cls, rect, score=None):
    """Draw a singel rect (if detection it may have score)
    :param ax: axis to draw on
    :param all_cls: all he classes
    :param cls: class for this rect
    :param rect: this rect
    :param score: probability score of this rect
    """
    if cls not in all_cls:
        idx = len(all_cls)
        color_idx = idx % (len(STANDARD_COLORS) / 2) * 2
        color = STANDARD_COLORS[color_idx]
        bgcolor = STANDARD_COLORS[color_idx + 1]
        all_cls[cls] = color, bgcolor
    else:
        color, bgcolor = all_cls[cls]
    left, top, right, bot = int_rect(rect)

    if score is not None:
        text = '%s: %.3f' % (cls, score)
    else:
        text = '{}'.format(cls)

    h0 = ax.add_patch(
        Rectangle((left, top),
                  right - left,
                  bot - top,
                  fill=False, edgecolor=color, linewidth=3)
    )
    if score:
        y = float(bot + 10)
    else:
        y = float(top - 10)
    h1 = ax.text(float(left), y, text,
                 color='black',
                 backgroundcolor=bgcolor,
                 bbox={
                     'facecolor': bgcolor,
                     'edgecolor': bgcolor,
                     'alpha': 0.5
                 })
    return h0, h1


def visualize(im, results, thresh=0.0, return_as_array=True, fig=None, path=None):
    """Visual debugging of a single prediction.
    :param im: image
    :type im: np.ndarray
    :param results: detection results to visualize
    :type results: list
    :param thresh: threshold to apply on top of the results
    :param return_as_array: if should return the visualization as an array
    :param fig: figure to draw on
    :type fig: Figure
    :param path: output path to save
    :type path: str
    """
    if fig is None:
        fig = Figure()
        canvas = FigureCanvas(fig)
    else:
        canvas = fig.canvas  # type: FigureCanvas
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = im[:, :, (2, 1, 0)]  # BGR to RGB

    all_cls = {}
    ax.imshow(im)
    handles = []
    for crect in results:
        score = crect.get('conf')
        if score is not None and score < thresh:
            continue
        cls = crect['class']
        handle = draw_rect(ax, all_cls, cls, crect['rect'], score=score)
        handles.append([handle, cls])

    if not handles:
        return ax, handles

    canvas.draw()
    if path:
        if op.splitext(path)[1].lower() not in _VALID_IMAGE_TYPES:
            path += ".jpg"  # supported format
        fig.savefig(path, bbox_inches='tight', pad_inches=0)

    if not return_as_array:
        return ax, handles

    width, height = fig.get_size_inches() * fig.get_dpi()
    width, height = int(width), int(height)
    return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)


def visualize_detections(exp, results, thresh=0.0):
    """Visual debugging of prediction.
    :type exp: mmod.experiment.Experiment
    :param results: detection results to visualize
    :type results: list
    :param thresh: threshold to apply on top of the results
    """
    vis = 0
    no_vis = 0
    total = len(exp.imdb)

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for key in exp.imdb:
        found = False
        all_cls = {}
        im = None
        uid = exp.imdb.uid(key)
        for crect in results:
            if crect['key'] != uid:
                continue
            score = crect['conf']
            if score < thresh:
                continue
            if im is None:
                im = exp.imdb[key]
                if im is None:
                    logging.error("Ignore {}".format(key))
                    break
                im = im[:, :, (2, 1, 0)]  # BGR to RGB
                ax.imshow(im)

            found = True
            draw_rect(ax, all_cls, crect['class'], crect['rect'], score=score)

        if found:
            canvas.draw()
            path = exp.vis_path(key)
            if op.splitext(path)[1].lower() not in _VALID_IMAGE_TYPES:
                path += ".jpg"  # supported format
            fig.savefig(path, bbox_inches='tight', pad_inches=0)
            ax.clear()  # clear for the next image
            vis += 1
            if vis % 100 == 0:
                logging.info("Visialzied: {} Ignored: {} out of: {} Last: {}".format(vis, no_vis, total, key))
        else:
            no_vis += 1
    return vis, no_vis, total


def exp_stat(exp):
    """Get the label metrics of an experiment
    :type exp: mmod.experiment.Experiment
    :rtype: pandas.DataFrame
    """
    sources = list(exp.imdb.iter_sources())
    cmapfile = exp.cmapfile
    if isinstance(cmapfile, list):
        index = cmapfile
    else:
        index = load_labelmap_list(exp.cmapfile)
    metrics = {source: np.zeros(len(index), dtype=np.uint32) for source in sources}
    metrics['total'] = np.zeros(len(index), dtype=np.uint32)
    for idx, label in enumerate(index):
        total = 0
        for key in exp.imdb.iter_label(label):
            total += 1
            for source in sources:
                if source == key[0]:
                    metrics[source][idx] += 1
        metrics['total'][idx] = total

    df = pd.DataFrame(metrics, index=index)
    return df
