from time import time
from warnings import warn

import torch


def check_each_shape(net: torch.nn.Sequential, input_tensor):
    for layer in net:
        input_tensor = layer(input_tensor)
        print(layer.__class__.__name__, 'outputshape:\t', x.shape)


class Timer(object):
    def __init__(self, describe: str = '', precision: int = 3):
        super(Timer, self).__init__()
        self.__start_count = False
        self.__time_start = None
        self._describe = describe
        self._precision = precision

    def __enter__(self):
        if self.__start_count:
            warn('timer is reset implicit', stacklevel=2)
        self.__start_count = True
        self.__time_start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time() - self.__time_start
        print('Timer:', self._describe, 'using {}sec'.format(round(duration, self._precision)))

    def start(self):
        if self.__start_count:
            warn('timer is reset implicit, using method "restart" instead.', stacklevel=2)
        self.__time_start = time()

    def stop(self, print_info: bool = True):
        duration = time() - self.__time_start
        if print_info:
            print(
                    'Timer:', self._describe,
                    'using {}sec'.format(round(duration, self._precision))
                    )
        return round(duration, self._precision)

    def restart(self):
        duration = time() - self.__time_start if self.__start_count else 0
        self.__start_count = True
        self.__time_start = time()
        return round(duration, self._precision)
