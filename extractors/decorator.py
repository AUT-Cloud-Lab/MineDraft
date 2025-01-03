from functools import wraps

extractor_functions = {}


def register_extractor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    extractor_functions[func.__name__] = wrapper
