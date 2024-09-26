# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

try:
    import megablocks
except ImportError:
    megablocks = None


def megablocks_is_available():
    return megablocks is not None

def assert_megablocks_is_available():
    assert megablocks_is_available(), (
        "megablocks is not available."
    )


ops = megablocks.ops if megablocks_is_available() else None
stk = megablocks.stk if megablocks_is_available() else None
sparse_act = megablocks.sparse_act if megablocks_is_available() else None
