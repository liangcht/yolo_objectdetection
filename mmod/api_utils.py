
def ltwh_to_ltrb(rect):
    """Convert rect from dictionary of left,top,width,height to list of left,top,right,bottom
    :type rect: dict
    :rtype: list
    """
    if not rect:
        return
    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
    return [x, y, x + w, y + h]


def convert_api(objects, child_rect=None):
    """Convert API results to detection results format
    :param objects: the list of objects
    :type objects: list[dict]
    :param child_rect: rectangle of the child to use if parent does not have one
    :type child_rect: list
    :return: list of detected objects
    :rtype: list[dict]
    """
    rects = []
    for o in objects:
        child_rect = ltwh_to_ltrb(o.get('rectangle')) or child_rect
        assert child_rect, "invalid object: {}".format(o)
        rect = {
            'class': o['object'],
            'conf': o['confidence'],
            'rect': child_rect
        }
        rects.append(rect)
        parent = o.get('parent')
        if parent:
            rects += convert_api([parent], child_rect=child_rect)
        # the sibling has to have its own rect
        child_rect = None

    return rects
