
def ltwh_to_ltrb(rect):
    """Convert rect from dictionary of left,top,width,height to list of left,top,right,bottom
    :type rect: dict
    :rtype: list
    """
    if not rect:
        return
    x, y = rect['x'] if 'x' in rect else rect['left'], rect['y'] if 'y' in rect else rect['top']
    w, h = rect['w'] if 'w' in rect else rect['width'], rect['h'] if 'h' in rect else rect['height']
    return [x, y, x + w, y + h]


def convert_api_od(objects, child_rect=None):
    """Convert OS API results to detection results format
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
            rects += convert_api_od([parent], child_rect=child_rect)
        # the sibling has to have its own rect
        child_rect = None

    return rects


def convert_api_celeb(objects):
    """Convert OS API results to detection results format
    :param objects: the list of objects
    :type objects: list[dict]
    :return: list of detected objects
    :rtype: list[dict]
    """
    rects = []
    for o in objects:
        roi = ltwh_to_ltrb(o.get('faceRectangle'))
        rect = {
            'class': o['name'],
            'conf': o['confidence'],
            'rect': roi
        }
        rects.append(rect)

    return rects
