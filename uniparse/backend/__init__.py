def init_backend(backend_name):
    if backend_name == "dynet":
        from uniparse.backend.dynet_backend import DynetBackend
        backend = DynetBackend()
    elif backend_name == "pytorch":
        from uniparse.backend.pytorch_backend import PyTorchBackend
        backend = PyTorchBackend()
    else:
        raise ValueError("backend doesn't exist: %s" % backend_name)
    return backend
