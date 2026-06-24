# Qwen3-VL-4B LLM backbone (Megatron MODEL_ARGS) for AgentFlow×MAT multimodal RL.
#
# Verified against HF Qwen/Qwen3-VL-4B-Instruct config.json -> text_config:
#   num_hidden_layers=36  hidden_size=2560  intermediate_size=9728
#   num_attention_heads=32  num_key_value_heads=8  head_dim=128
#   vocab_size=151936  rms_norm_eps=1e-6  rope_theta=5000000  tie_word_embeddings=true
#
# The text backbone is the same Qwen3 dense arch as scripts/models/qwen3-4B.sh; the
# ONLY difference is rope_theta (5e6 here vs 1e6). The vision tower is NOT described
# here — slime's qwen3_vl support builds it from the HF checkpoint.
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-5000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
