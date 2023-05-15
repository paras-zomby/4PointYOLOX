import sys
from functools import wraps
from typing import Union, TextIO

from ptflops import get_model_complexity_info
from torchstat import stat


def redirect_output(new_out: Union[str, bytes, TextIO] = sys.stderr):
    def decorator(func):
        @wraps(func)
        def decorated_func(*args, **kwargs):
            out = open(new_out, "w", encoding='utf-8') \
                if isinstance(new_out, str) or \
                   isinstance(new_out, bytes) else new_out
            stdout = sys.stdout
            sys.stdout = out
            func_out = func(*args, **kwargs)
            sys.stdout = stdout
            if isinstance(new_out, str) or isinstance(new_out, bytes):
                out.close()
            return func_out

        return decorated_func

    return decorator


def get_model_complex(model, input_size):
    stat(model, (3, *input_size))


def get_total_params_num(model):
    print('parameters: ', sum(param.numel() for param in model.parameters()))


def get_model_flops(model, input_size):
    macs, params = get_model_complexity_info(
            model, (3, *input_size), as_strings=True,
            print_per_layer_stat=True, verbose=True
            )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
