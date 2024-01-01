from functools import wraps

extractor_scripts = {}


def register_extractor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    extractor_scripts[func.__name__] = wrapper
