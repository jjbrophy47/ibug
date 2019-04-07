import time
from functools import wraps


def timing(f):

    @wraps(f)
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()

        print('* %s function took [%.3fs]' % (f.__name__, time2 - time1))
        return ret

    return wrap
