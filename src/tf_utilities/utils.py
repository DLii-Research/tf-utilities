def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def str_to_bool(s):
    return s.strip().lower() in {'1', 't', 'true', 'y', 'yes'}
