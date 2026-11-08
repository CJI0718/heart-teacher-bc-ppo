# Sources and implementation boundaries

## Official environment

- [funny-rl/Heart](https://github.com/funny-rl/Heart)
- Pinned commit: `ad122dba1e1d4c0aaee9bee8846ceba60ed680fe`
- [Observation and submission contract](https://github.com/funny-rl/Heart/blob/ad122dba1e1d4c0aaee9bee8846ceba60ed680fe/docs/leaderboard.md)
- Environment license: Apache-2.0. Installed as a dependency, not vendored here.

No instructor slides, private course PDF, personal notes, credentials, or paper PDF are included.

## Hearts feature construction

Nathan R. Sturtevant and Adam M. White (2007), **Feature Construction for Reinforcement Learning in Hearts**.

- [Author PDF](https://www.cs.du.edu/~sturtevant/papers/heartsfeatures.pdf)
- [Publication](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_11)
- Relevant parts: §1.4 linear approximation, §2.2 TD λ=.75, §2.3–2.4 components, §2.5 feature integration, §2.6 imperfect-information evaluation.

The paper learns perfect-information state evaluations and combines them with max-n / sampled-hand search. It selects 2,000 hearts features and 10,000 queen features for full-game retraining. Shooting the moon is disabled during §2.5 training and enabled during §2.6 evaluation.

This project borrows **component prediction → feature selection → integrated retraining**. Changes are explicit:

- Observable features only, no hidden hands or perfect-information search.
- Action-conditioned linear cost and semi-gradient Sarsa(λ), not a verbatim implementation of the paper's backward state-value updates.
- A small 436-feature candidate basis rather than the paper's high-order exhaustive feature set.
- Coefficient/activity screening rather than claiming the original feature-selection algorithm is reproduced.
- Self/opponent moon auxiliary prediction tasks, followed by effective-score teacher training.
- Behavioral cloning into attention, then PPO. The 2007 paper did not implement this pipeline.

## Standard methods

- Sutton and Barto (2018), **Reinforcement Learning: An Introduction**, 2nd edition: Sarsa, eligibility traces, function approximation. [Author site](http://incompleteideas.net/book/the-book-2nd.html)
- Schulman et al. (2017), **Proximal Policy Optimization Algorithms**. [Paper](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2016), **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. [Paper](https://arxiv.org/abs/1506.02438)
- [JAX documentation](https://docs.jax.dev/en/latest/), [Optax documentation](https://optax.readthedocs.io/en/latest/)

These are algorithmic references. No published pretrained model is used. KL anchors, curricula, opponent pools, GO-MCTS and other papers' extensions are out of scope for this version.
