
def wrapped_call(func, unnamed_args, named_args):  # without star
    """
    is a wrapper function that allows an efficient call of function by passing a list of unnamed_args and dict of
    named_args

    Returns: result of call
    """
    return func(*unnamed_args, **named_args)