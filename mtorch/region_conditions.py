"""Helper module to define conditions on bounding boxes"""


def is_valid(bbox):
    return _width(bbox) and _height(bbox)


def _area(bbox):
    return _width(bbox) * _height(bbox)


def _width(bbox):
    return bbox[2] - bbox[0]


def _height(bbox):
    return bbox[3] - bbox[1]


class BboxCondition(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, bbox):
        if not is_valid(bbox):
            return False
        return self._is_condition_valid(bbox)

    def _is_condition_valid(self, bbox): 
        raise NotImplementedError("This is an abstract class")
 

class HasAreaAbove(BboxCondition):

    def __init__(self, area_thresh):
        super(HasAreaAbove, self).__init__(area_thresh)

    def __call__(self, bbox): 
        return super(HasAreaAbove, self).__call__(bbox)
   
    def _is_condition_valid(self, bbox):       
        return _area(bbox) > self.thresh
                

class HasWidthAbove(BboxCondition):

    def __init__(self, width_thresh=8):
        super(HasWidthAbove, self).__init__(width_thresh)

    def __call__(self, bbox): 
        return super(HasWidthAbove, self).__call__(bbox)
   
    def _is_condition_valid(self, bbox):       
        return _width(bbox) > self.thresh


class HasHeightAbove(BboxCondition):

    def __init__(self, height_thresh=8):
        super(HasHeightAbove, self).__init__(height_thresh)

    def __call__(self, bbox): 
        return super(HasHeightAbove, self).__call__(bbox)
   
    def _is_condition_valid(self, bbox):       
        return _height(bbox) > self.thresh



