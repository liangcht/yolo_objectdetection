import numpy as np

BBOX_DIM = 4

class Labeler(object):
    """
    Creates labels in a format of top left bottom right corners of bounding box
    Attaches class per each label
    """

    def __init__(self):
        """Constructor of Labeler Class"""
        pass
    
    def __call__(self, truth_list, cmap, filter_difficult=True):
        """
        Constructs bounding boxes according to the following format:
        x for left, y for top, x for right, y for bottom, class
        :param truth_list: bounding boxes
        :param cmap: the map the converts between the class string labels
        and corresponding numeric labels
        :param filter_difficult: boolean, if true filters difficult labels,
        if false retains all labels
        :return: number of boxes x 5 numpy array of float32
        """
        return self.create_bounding_boxes(truth_list, cmap, filter_difficult)

    @staticmethod
    def create_bounding_boxes(truth, cmap, filter_difficult):
        """
        Create bounding boxes
        :param truth: bounding boxes
        :param cmap: class to numeric value conversion map
        :param filter_difficult: boolean, if true filters difficult labels,
        if false retains all labels
        :return: number of boxes x 5 numpy array of float32
        """
        length = len(truth)
        bboxs = np.zeros(shape=(length, BBOX_DIM + 1), dtype="float32")
        last_valid_box = 0
        for bbox in truth:
            if filter_difficult and bbox.get('diff', 0) == 1:
                continue
            bboxs[last_valid_box, :BBOX_DIM] = [float(val) for val in bbox['rect']]
            bboxs[last_valid_box, BBOX_DIM] = cmap.index(bbox['class'])
            last_valid_box += 1
        bboxs = bboxs[:last_valid_box, :]
        return bboxs


