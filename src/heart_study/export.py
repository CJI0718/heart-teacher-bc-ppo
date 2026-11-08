"""The same observation contract as the official host, with finite two-head output."""

import jax
from .environment import fixed_pass_logits
from .observations import encode
from .model import batched_forward


def submission_policy(params, spec, model_config):
    def policy(observation):
        encoded = jax.vmap(encode)(observation)
        features = jax.vmap(spec.__call__)(observation)
        play, _ = batched_forward(params, encoded, features, model_config)
        return jax.vmap(fixed_pass_logits)(observation), play

    return policy


def export_checkpoint(checkpoint, output):
    import heart
    from .training import load_student

    params, spec, mc, _ = load_student(checkpoint)
    return heart.save_submission(
        submission_policy(params, spec, mc),
        output,
        name="heart-linear-teacher-attention",
        description="Linear teacher BC followed by PPO",
    )
