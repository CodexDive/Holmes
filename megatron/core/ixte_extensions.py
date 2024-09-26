_USE_IXTE = False
try:
    from transformer_engine import ixte_version, te_version
    from transformer_engine.pytorch import (
        get_logits_linear_func,
        vocab_parallel_cross_entropy,
    )
    from transformer_engine.common.optimizing_config import (
        get_embedding_tp_overlap_size,
    )
    from transformer_engine.common import get_opt_config
    from transformer_engine.common import initialize_ixte

    _USE_IXTE = True
except ImportError:
    pass
