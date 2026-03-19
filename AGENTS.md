# Notes

- Collaboration fallback -> if an agent hits an environment limitation, harness issue, missing access, or any blocker it cannot safely resolve, it should ask the user for help instead of guessing or stalling -> keeps loops moving when the local setup is the bottleneck.
- Shared ideation backlog -> `ideas.md` is the living ranked idea backlog and experiment ledger; agents should update it when a new idea appears, when rankings change, and whenever a tried idea gets a result -> keeps the current strategy and evidence in one place.
- Local Python setup -> the repo is on `C:`, but the working Python environment is at `D:\venvs\parameter-golf`; prefer that venv for installs and runs because `C:` space is tight -> prevents repeated CUDA install failures from disk pressure.
- Windows local launch -> for single-GPU local loops prefer `python train_gpt.py` over `torchrun`; the Windows launcher path can trip over libuv/rendezvous issues that do not matter for the 1-GPU smoke workflow -> avoids wasting time on launcher-specific failures.
- Git checkpoints -> make descriptive checkpoint commits after substantial progress and push when the result is worth preserving remotely -> keeps rollback points available during long experiment loops.
- Local dev environment -> primary machine is Windows 11 with an RTX 4060 -> use it for CUDA smoke tests and iteration, not as a proxy for 8xH100 leaderboard performance.
- Submission scoring -> final metric is post-quant `final_int8_zlib_roundtrip_exact val_bpb`, not the pre-quant validation loss -> optimizing a model that breaks after int8+zlib can look good mid-run but still fail the leaderboard metric.
- Artifact accounting -> the 16,000,000-byte cap includes counted code plus compressed model bytes, and challenge submissions are expected to keep counted code in `train_gpt.py` while record PRs add a new folder under `records/` -> changing only root trainer code is not a valid final submission shape.
- Evaluation/data -> validation is tokenizer-agnostic BPB on the fixed first-50k-doc `fineweb_val_*` split; tokenizer changes are allowed but scrutinized and must preserve correct byte accounting -> tokenizer bugs can create invalid wins.
- Exporter direction -> row-centered int8 export is width-sensitive: it consistently helped the local `9/3 @ 1024` branch for ~36 KB extra compressed size, but slightly hurt the current best `9/3 @ 896` branch, while grouped per-row scales were mostly noise -> treat centered export as a recovery tool for wider models, not a blanket default.
- Research direction -> multiple independent research passes converged on rotation/incoherence-based quantization stabilization as the strongest new idea, with scale-reparameterization second, but the fixed Hadamard exporter probe underperformed; the first exact scale-reparameterization pass shows that `v ↔ proj` equalization helps both `896` and `1024` slightly while `relu^2` MLP equalization is the unstable part -> if this family is revisited, focus on attention value/projection equalization rather than bundled MLP+attention transforms.

## Main Goals

- Official challenge target -> minimize held-out loss on the fixed FineWeb dataset while staying within a strict `16 MB` artifact limit and a `10 minute` training budget on `8x H100s` -> this is the actual optimization problem from the OpenAI challenge page and repo.
- Real win condition -> lower the final post-roundtrip metric `final_int8_zlib_roundtrip_exact val_bpb` on a leaderboard-safe run -> local pre-quant improvements alone are not wins.
- Good wins -> architecture, quantization, tokenizer, or systems changes that improve final post-roundtrip `val_bpb`, preserve artifact viability, and plausibly scale to the real challenge budget -> these are the changes worth spending time on.
- Weak wins -> local-only speedups, tiny pre-quant gains that vanish after quantization, or tweaks that only help the Windows smoke loop -> do not confuse iteration convenience with challenge progress.
- Local loop purpose -> use the RTX 4060 setup to reject bad ideas quickly and identify promising ones for later serious runs -> treat local numbers as directional, not authoritative.
