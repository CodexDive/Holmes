# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
# TODO: @aoyulong need to choose RMSNorm impl
# from .rms_norm import RMSNorm
from .fused_rms_norm import MixedFusedRMSNorm as RMSNorm

from .bert_model import BertModel
from .gpt_model import GPTModel
from .t5_model import T5Model
from .llama_model import LLaMAModel
from .chatglm_model import ChatGLMModel
from .language_model import get_language_model
from .module import Float16Module
