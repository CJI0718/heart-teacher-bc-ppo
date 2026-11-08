# Verification record

This directory records **construction checks, not trained-agent performance**.

All checks use Python 3.12 and the pinned official HEART revision. The tests cover:

- zero-reward deal termination and rollout-edge bootstrapping;
- GAE and PPO clipping hand calculations;
- Sarsa target uses the actual next action;
- feature-set union without averaging component weights;
- public history order, reconstructed seats, and confirmed voids;
- hand permutation equivariance and critic invariance;
- finite logits, masking, BC/PPO loss and gradient computation;
- official PASS `[batch,286]` / PLAY `[batch,13]` output;
- default training commands perform no training;
- array-only checkpoint round trip and moon-adjusted terminal labels.

`smoke_check.json` records one deal with **frozen random initial weights**.
`export_check.json` records compilation of **random initial weights**, not a trained submission.
No optimizer step, Sarsa parameter update, training epoch, or policy improvement run is executed during these checks.

The actual training loops and optimizer steps remain execution-unverified until training is separately authorized. Passing the above checks does not establish convergence, competitive strength, or GPU throughput.
