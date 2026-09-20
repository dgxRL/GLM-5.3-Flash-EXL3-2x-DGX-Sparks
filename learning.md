# Startup recovery — 2026-09-20

## Observed failures

- `./start.sh` stopped at the worker memory preflight: only 22.80 GiB was
  available, versus 103.44 GiB requested plus 2 GiB headroom.
- The existing head container had exited while its worker remained running.
  The head log reported that the configured 850,000-token context needed
  13.46 GiB of KV cache, but only 10.96 GiB was available. CUDA graph memory
  estimation reserved 1.42 GiB. InstantTensor was the active loader.
- Worker NCCL/TCPStore broken-pipe messages followed the head's exit; the
  surviving worker retained model memory and blocked a new plain start.

## Changes

Only local `.env` settings were changed; application code was not modified.

```bash
LOAD_FORMAT=
SKIP_PULL=1
```

- An explicitly empty `LOAD_FORMAT` selects vLLM auto / standard safetensors
  loading. Removing the setting altogether would enable InstantTensor again
  because the image tag contains `instanttensor`. The comparison below confirms
  that this restores enough KV budget on this pair.
- `SKIP_PULL=1` retains the existing repository-built image for reproducible
  restarts. Recipe-stamp changes still trigger automatic rebuilds. The image
  used for this run is
  `sha256:48b65ea895a6d1fff51b067c54135f98d713fca873e97625648ada94cb56c457`.
- `./start.sh restart` removed and recreated the two failed model containers,
  freeing the worker's memory. It did not stop the separate sparkDash service.
- Preserved `MAX_MODEL_LEN=850000`, `GPU_MEM_UTIL=0.85`, `CG_ESTIMATE=1`,
  DFlash2 k=7, and tensor parallelism across both nodes.

## Validation

- `.env` passes `bash -n`.
- Restart preflight passed with 117.11 GiB available on the head and
  117.99 GiB on the worker.
- The launcher's GPU EXL3 overlay self-check passed (`logs/overlay-verify.log`).
- Both ranks passed the 850k KV-capacity check; the effective reported pool is
  963,439 tokens / 1.13x maximum concurrency at 850k, limited by the worker.
- `./start.sh restart` exited successfully. Health passed after 630 seconds
  of launcher polling (startup patches, loading, profiling, and initialization).
- Post-ready shape warmup passed **24/24 requests in 62 seconds**, including
  a 65,536-token prefill and concurrency through four requests.
- A real `/v1/chat/completions` request for `17 + 25` returned `42` with
  `finish_reason=stop`; `/health` returned HTTP 200 afterward.
- These are startup and smoke checks, not a full 850k-context stress test.

## Why InstantTensor failed here

This is a reported upstream problem, not evidence that this particular pair
is uniquely broken. The default changed on **2026-09-16**, four days before
this recovery (`9c1c528`, `8f44c05`, CHANGELOG 1.4.0). Older August/early-September
benchmark receipts do not validate the new loader default.

The same repository has an open report, [issue #204](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/204),
describing the exact 850k / 0.85 failure. Its same-image comparison recovered
4.43 GiB by clearing `LOAD_FORMAT`; two other users confirmed the startup
failure. The original [PR #200](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/200)
does report a successful TP=2 health check with InstantTensor and
`NCCL_NCHANNELS=8`, but does not provide the complete memory configuration in
its test plan. Our run leaves channel selection automatic. This difference
has not been isolated and is not established as the cause.

Our measurements (same local image, 850k context, utilization 0.85,
CUDA graph estimation enabled, DFlash2, identical 82.05 GiB model allocation):

| Measurement | InstantTensor, failed previous launch | Standard safetensors, recovery |
| --- | ---: | ---: |
| Head available KV | 10.96 GiB | 18.25 GiB |
| Worker available KV | 10.89 GiB | 15.26 GiB |
| Head CUDA graph estimate | 1.42 GiB | 1.46 GiB |
| Worker CUDA graph estimate | 1.32 GiB | 1.59 GiB |
| Head model-loading time | 45.51 s | 332.46 s |
| Worker model-loading time | 37.89 s | 136.88 s |
| Minimum KV needed for one 850k request | 13.46 GiB | 13.46 GiB |

The limiting worker gains **4.37 GiB**, closely matching #204. The head gains
7.29 GiB; do not attribute the entire larger difference to the loader. On
GB10 the installed vLLM memory profiler uses host `MemAvailable`, so changing
host/cache state between launches also affects its measured budget. This was
one recovery comparison against a preserved failed launch, not a repeated
A/B/A experiment under identical OS memory state.

The installed code explains what the failure does and does not mean:

- `start.sh:261` tests whether `LOAD_FORMAT` is **set**, not whether it is
  nonempty. Unset + an `instanttensor` image name selects InstantTensor;
  explicitly empty survives, and the generated command omits `--load-format`.
  vLLM then selects its default loader; this run logged `load_format=auto` and
  loaded the safetensors shards normally. Both paths read the same weights.
- vLLM computes the automatic KV allowance as the requested memory budget
  minus profiled non-KV consumption minus the CUDA-graph estimate. A loader
  can finish successfully yet leave insufficient *profiled KV allowance*.
  That is what happened here; the original fatal error was not a shard-read
  failure. Disabling the graph estimate alone would recover only about
  1.3–1.4 GiB in that run, less than the 2.57 GiB worker shortfall.
- InstantTensor 0.2.0 uses a distributed GPU ring buffer, pinned host staging,
  NCCL transfers, and `copy=True` tensor clones in vLLM. Its direct-I/O default
  is 8 MiB chunks and `512/world_size` in-flight operations (256 at TP=2),
  reduced when memory is scarce. The failed log shows reductions to 239 and
  172. It limits the logical device buffer to half of CUDA-reported free
  memory, taking the minimum across ranks.
- **Do not call the KV difference a proven buffer leak.** The tagged native
  implementation explicitly frees the GPU ring and unregisters/frees host
  staging on close; pinned-buffer caching defaults to off. The exact split
  between persistent communication/allocator state and UMA accounting effects
  has not been measured. The evidence establishes the loader-dependent
  budget regression and a working workaround, not its lowest-level cause.

A separate open [issue #205](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/205)
explains another failure on fresh installs: rsync fills reclaimable Linux page
cache, which preflight counts as available but InstantTensor's
`cudaMemGetInfo` budget counts as used. Its buffer constructor can reject the
load before reading weights. That explains why cold/warm starts and machines
can differ, but our preserved failure completed weight loading, so #205's
constructor failure was not observed in this run. The issue also reports a
surviving worker blocking the next start, matching our recovery situation.

Retaining standard loading costs several minutes at startup but preserves the
850k context and 0.85 memory setting. Raising utilization to 0.88 is a reported
alternative in #204; it reduces host headroom and was not tested or applied
here. Retuning InstantTensor I/O or NCCL is also untested on this pair.

Implementation references: installed
`vllm/model_executor/model_loader/weight_utils.py`,
`vllm/v1/worker/gpu_worker.py`, `vllm/utils/mem_utils.py`, and
`instanttensor/_impl.py`; [InstantTensor v0.2.0 native buffer lifecycle](https://github.com/scitix/InstantTensor/blob/v0.2.0/csrc/loader_common.cpp)
and [buffer-cache default](https://github.com/scitix/InstantTensor/blob/v0.2.0/csrc/instant_tensor/common.hpp).
Local versions: vLLM `0.1.dev20051+g487ecf187`, InstantTensor `0.2.0`,
NCCL `2.30.7`; head driver `580.173.02`, kernel `6.17.0-1031-nvidia`.

## Operational notes

- Use `./start.sh restart` to recover a failed pair with a surviving rank.
  A plain `./start.sh` checks memory before replacing existing containers.
- The original failed-container logs were saved before replacement in
  `/tmp/glm53-before-fix-head.log` and `/tmp/glm53-before-fix-worker.log`.
  Before/after copies are archived in `logs/startup-recovery-20260920/` as
  `before-head.log`, `before-worker.log`, `after-head.log`, and `after-worker.log`.
- `.env` is Git-ignored, so the local configuration changes do not appear in
  normal `git diff`. This document records the settings to reproduce them.
