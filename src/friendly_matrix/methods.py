__all__ = ['take_A', 'take', 'get', 'set', 'copy']


def take_A(friendly, *args, **kwargs):
	return friendly.take_A(*args, **kwargs)


def take(friendly, *args, **kwargs):
    return friendly.take(*args, **kwargs)


def get(friendly, *args, **kwargs):
    return friendly.get(*args, **kwargs)


def set(friendly, val, *args, **kwargs):
    return friendly.set(val, *args, **kwargs)


def copy(friendly):
    return friendly.copy()