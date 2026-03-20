This record captures the current best local cheap-cloud `SP-4096` smoke candidate.

Important scope note:
- this is **not** a submission-grade `8xH100 / 10 minute` result
- it is a `1x RTX 3090` capped-validation smoke used to validate branch shape, artifact viability, and exact post-quant behavior before spending a real submission shot

Trainer changes in this snapshot:
- current repository `train_gpt.py` snapshot copied into this record folder
- `SP-4096` dataset and tokenizer built from the local docs re-export on Runpod
- doc-isolated validation path enabled
- lower-LR / longer-warmdown schedule applied

Configuration:
- Tokenizer/data: `VOCAB_SIZE=4096`, `fineweb10B_sp4096`
- Layout: `NUM_LAYERS=9 NUM_SHARED_BLOCKS=3 NUM_SHARED_MLPS=3 MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1664`
- Context: `TRAIN_SEQ_LEN=1024`, `EVAL_STRIDE_TOKENS=64`, `EVAL_DOC_ISOLATED=1`
- Batching/smoke cap: `TRAIN_BATCH_TOKENS=32768`, `ITERATIONS=20`, `VAL_MAX_TOKENS=1048576`
- LR schedule: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500`

Command (track-relevant params):
```bash
RUN_ID=sp4096_smoke_fix1 \
DATA_PATH=/workspace/pg-sp4096-export/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=/workspace/pg-sp4096-export/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
NUM_LAYERS=9 \
NUM_SHARED_BLOCKS=3 \
NUM_SHARED_MLPS=3 \
MODEL_DIM=640 \
MLP_HIDDEN=1664 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=32768 \
ITERATIONS=20 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=10 \
MAX_WALLCLOCK_SECONDS=180 \
EVAL_STRIDE_TOKENS=64 \
EVAL_DOC_ISOLATED=1 \
VAL_MAX_TOKENS=1048576 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
DISABLE_COMPILE=1 \
python3 train_gpt.py
```

Key metrics (from `train.log`):
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:2.84400507`
- Exact printed loss: `final_int8_zlib_roundtrip_exact val_loss:6.46848655`
- Pre-roundtrip smoke eval at stop: `val_bpb:2.8443`
- Train time at stop: `9624ms` (`step_avg:481.19ms`)
- Peak memory: `1547 MiB allocated`, `1632 MiB reserved`
- Serialized model int8+zlib: `6021764 bytes`
- Code size: `101448 bytes`
- Total submission size int8+zlib: `6123212 bytes`

Interpretation:
- this is the best current `SP-4096` smoke branch on the cheap-cloud rung
- later smoke branches tested against it and lost:
  - `MODEL_DIM=768 MLP_HIDDEN=2048`
  - `INT8_KEEP_TOK_EMB_FP16=1`
  - `TRAIN_SEQ_LEN=2048`

Included files:
- `train_gpt.py` (code snapshot used by the smoke branch)
- `train.log` (captured exact log excerpt from the successful cloud smoke)
- `submission.json` (metadata for this smoke record)
