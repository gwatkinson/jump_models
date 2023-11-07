"""Contains utility functions for layers."""


def _get_layer(pl_module, layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    layer = pl_module
    for name in layer_names:
        layer = getattr(layer, name)
    return layer


def _has_layer(pl_module, layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    layer = pl_module
    for name in layer_names:
        if not hasattr(layer, name):
            return False
        layer = getattr(layer, name)
    return True
