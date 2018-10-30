
class CaffeLoader(object):
    def __init__(self, inputs):
        """Simple loader to wrap a nn module
        The module is supposed to forward a data layer
        :type inputs: torch.nn.Module
        """
        self._inputs = inputs

    def __iter__(self):
        yield self._inputs()
