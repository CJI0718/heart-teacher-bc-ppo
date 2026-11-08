"""Training stages. Defined here, but never run at import or by verification.

All loops deliberately use one environment for a readable reference. They are
not a GPU-throughput implementation. No anchor, curriculum, replay or league.
"""

from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpoints import save_arrays, load_arrays, pack_params, unpack_params
from .config import TrainConfig, ModelConfig, COMPONENTS, HEART_COMMIT
from .environment import PlayRunner, rule_play
from .features import FeatureSpec, select_union
from .teacher import component_update, sarsa_update, epsilon_action, greedy_action
from .observations import encode, stack_encoded, Encoded
from .model import init_params, log_probabilities, compiled_observation_policy
from .losses import Batch, bc_loss, ppo_loss, gae


def meta(stage, config, spec, **extra):
    return {
        "format_version": 1,
        "stage": stage,
        "heart_commit": HEART_COMMIT,
        "training_executed": True,
        "config": config.to_dict(),
        "features": spec.to_dict(),
        "checkpoint_kind": "stage_handoff_weights",
        **extra,
    }


def train_components(output, deals, config=TrainConfig()):
    runner, spec = PlayRunner(config.seed, config.opponents), FeatureSpec()
    phi_fn = jax.jit(spec.__call__)
    rule = jax.jit(rule_play)
    weights = jnp.zeros((4, len(spec.names)))
    traces = jnp.zeros_like(weights)
    update = jax.jit(
        lambda w, e, x, n, c, t: component_update(
            w, e, x, n, c, t, config.teacher_alpha, config.gamma, config.teacher_lambda
        )
    )
    activity, target_sums, steps, finished = np.zeros(len(spec.names)), np.zeros(4), 0, 0
    key = jax.random.key(config.seed + 1)
    key, action_key = jax.random.split(key)
    action = int(rule(runner.obs, action_key))
    while finished < deals:
        phi = phi_fn(runner.obs)[action]
        _, _, terminal, labels = runner.step(action)
        key, action_key = jax.random.split(key)
        next_action = int(rule(runner.obs, action_key))
        next_phi = jnp.zeros_like(phi) if terminal else phi_fn(runner.obs)[next_action]
        # Auxiliary costs are normalized for comparable selection scores.
        target = np.zeros(4) if not terminal else labels / np.array([13, 13, 1, 1])
        weights, traces, _ = update(weights, traces, phi, next_phi, target, terminal)
        activity += np.asarray(phi) ** 2
        steps += 1
        if terminal:
            target_sums += labels
            finished += 1
        action = next_action  # the actual behavior action used in the TD target
    save_arrays(
        output,
        {"weights": weights, "activity": activity / steps},
        meta(
            "components",
            config,
            spec,
            deals=finished,
            steps=steps,
            component_names=COMPONENTS,
            raw_target_sums=target_sums.tolist(),
            label_scales=[13, 13, 1, 1],
            behavior="fixed official medium rule",
        ),
    )


def select_features(checkpoint, output, top_per_task=32):
    arrays, metadata = load_arrays(checkpoint)
    if metadata["stage"] != "components" or not metadata.get("training_executed"):
        raise ValueError("a trained component checkpoint is required")
    spec = FeatureSpec.from_dict(metadata["features"])
    selected, groups = select_union(arrays["weights"], arrays["activity"], spec, top_per_task)
    import json
    from pathlib import Path

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(
            {
                "features": selected.to_dict(),
                "source_stage": "components",
                "per_task_indices": groups,
                "component_names": COMPONENTS,
                "method": "absolute coefficient times RMS activity; base features retained",
                "raw_target_sums": metadata["raw_target_sums"],
            },
            f,
            indent=2,
        )


def read_features(path):
    from pathlib import Path
    import json

    return FeatureSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))["features"])


def train_teacher(features_path, output, deals, config=TrainConfig()):
    runner, spec = PlayRunner(config.seed, config.opponents), read_features(features_path)
    phi_fn = jax.jit(spec.__call__)
    weights, trace = jnp.zeros(len(spec.names)), jnp.zeros(len(spec.names))
    choose = jax.jit(lambda w, p, legal, k: epsilon_action(w, p, legal, k, config.teacher_epsilon))
    update = jax.jit(
        lambda w, e, x, n, c, t: sarsa_update(
            w, e, x, n, c, t, config.teacher_alpha, config.gamma, config.teacher_lambda
        )
    )
    key = jax.random.key(config.seed + 2)
    key, action_key = jax.random.split(key)
    action = int(choose(weights, phi_fn(runner.obs), runner.obs.play_action_mask, action_key))
    steps, finished = 0, 0
    while finished < deals:
        phi = phi_fn(runner.obs)[action]
        _, reward, terminal, _ = runner.step(action)
        key, action_key = jax.random.split(key)
        next_action = int(
            choose(weights, phi_fn(runner.obs), runner.obs.play_action_mask, action_key)
        )
        next_phi = jnp.zeros_like(phi) if terminal else phi_fn(runner.obs)[next_action]
        weights, trace, _ = update(weights, trace, phi, next_phi, -reward, terminal)
        steps += 1
        finished += int(terminal)
        action = next_action
    save_arrays(
        output,
        {"weights": weights},
        meta(
            "teacher",
            config,
            spec,
            deals=finished,
            steps=steps,
            algorithm="semi-gradient Sarsa(lambda)",
            target="effective deal penalty",
        ),
    )


def load_teacher(path):
    arrays, metadata = load_arrays(path)
    if metadata["stage"] != "teacher" or not metadata.get("training_executed"):
        raise ValueError("a trained integrated linear teacher is required")
    spec = FeatureSpec.from_dict(metadata["features"])
    if arrays["weights"].shape != (len(spec.names),):
        raise ValueError("teacher shape does not match its feature specification")
    return jnp.asarray(arrays["weights"]), spec, metadata


def collect_demos(teacher_path, output, deals, config=TrainConfig()):
    """Frozen teacher actions, not initial rule-policy actions, become BC labels."""
    weights, spec, teacher_meta = load_teacher(teacher_path)
    runner = PlayRunner(config.seed, config.opponents)
    choose = jax.jit(lambda o: greedy_action(weights, spec(o), o.play_action_mask))
    observations, features, actions, finished = [], [], [], 0
    while finished < deals:
        obs = runner.obs
        action = int(choose(obs))
        observations.append(encode(obs))
        features.append(spec(obs))
        actions.append(action)
        _, _, terminal, _ = runner.step(action)
        finished += int(terminal)
    encoded = stack_encoded(observations)
    arrays = {"obs_" + name: getattr(encoded, name) for name in Encoded._fields}
    arrays.update(features=jnp.stack(features), actions=np.array(actions, np.int32))
    m = meta(
        "demonstrations",
        config,
        spec,
        deals=deals,
        decisions=len(actions),
        teacher_stage=teacher_meta["stage"],
        labels="frozen linear teacher legal argmin",
    )
    m["training_executed"] = False  # this stage collects examples without fitting
    save_arrays(output, arrays, m)


def load_demos(path):
    arrays, metadata = load_arrays(path)
    if metadata["stage"] != "demonstrations":
        raise ValueError("expected demonstrations")
    encoded = Encoded(*(jnp.asarray(arrays["obs_" + name]) for name in Encoded._fields))
    actions = jnp.asarray(arrays["actions"])
    if len(actions) == 0 or not np.all(np.asarray(encoded.legal)[np.arange(len(actions)), actions]):
        raise ValueError("demonstrations must contain legal action labels")
    return encoded, jnp.asarray(arrays["features"]), actions, metadata


def optimizer(config):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate)
    )


def train_bc(demos_path, output, epochs, config=TrainConfig(), model_config=ModelConfig()):
    obs, features, actions, metadata = load_demos(demos_path)
    spec = FeatureSpec.from_dict(metadata["features"])
    first = jax.tree_util.tree_map(lambda x: x[0], obs)
    key = jax.random.key(config.seed + 3)
    key, init_key = jax.random.split(key)
    params = init_params(init_key, first, len(spec.names), model_config)
    opt = optimizer(config)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s, o, f, a):
        loss, grad = jax.value_and_grad(lambda pp: bc_loss(pp, o, f, a, model_config))(p)
        updates, s = opt.update(grad, s, p)
        return optax.apply_updates(p, updates), s, loss

    losses = []
    for _ in range(epochs):
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, len(actions)))
        for start in range(0, len(order), config.batch_size):
            ix = order[start : start + config.batch_size]
            params, opt_state, loss = step(
                params,
                opt_state,
                jax.tree_util.tree_map(lambda x: x[ix], obs),
                features[ix],
                actions[ix],
            )
            losses.append(float(loss))
    save_arrays(
        output,
        pack_params(params),
        meta(
            "bc",
            config,
            spec,
            epochs=epochs,
            model_config=asdict(model_config),
            updates=len(losses),
            last_loss=losses[-1],
        ),
    )


def load_student(path):
    arrays, m = load_arrays(path)
    if m["stage"] not in ("bc", "ppo") or not m.get("training_executed"):
        raise ValueError("expected trained attention BC or PPO checkpoint")
    return (
        unpack_params(arrays),
        FeatureSpec.from_dict(m["features"]),
        ModelConfig(**m["model_config"]),
        m,
    )


def collect_rollout(runner, params, spec, model_config, key, steps, config=TrainConfig()):
    """New on-policy data at every PPO iteration; no parameter update here."""
    evaluate = compiled_observation_policy(spec, model_config)
    observations, features, actions, logps, values = [], [], [], [], []
    rewards, terminals, next_values = [], [], []
    for _ in range(steps):
        obs = runner.obs
        logits, value = evaluate(params, obs)
        key, action_key = jax.random.split(key)
        logp = log_probabilities(logits, obs.play_action_mask)
        action = int(jax.random.categorical(action_key, logp))
        observations.append(encode(obs))
        features.append(spec(obs))
        actions.append(action)
        logps.append(logp[action])
        values.append(value)
        _, reward, terminal, _ = runner.step(action)
        next_value = 0.0 if terminal else float(evaluate(params, runner.obs)[1])
        rewards.append(reward)
        terminals.append(terminal)
        next_values.append(next_value)
    adv, ret = gae(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(next_values),
        jnp.asarray(terminals),
        config.gamma,
        config.gae_lambda,
    )
    normalized = (adv - jnp.mean(adv)) / jnp.maximum(jnp.std(adv), 1e-8)
    batch = Batch(
        stack_encoded(observations),
        jnp.stack(features),
        jnp.asarray(actions),
        jnp.asarray(logps),
        jax.lax.stop_gradient(normalized),
        jax.lax.stop_gradient(ret),
    )
    return batch, key, {"deals": sum(terminals), "reward_sum": sum(rewards)}


def train_ppo(bc_path, output, iterations, config=TrainConfig()):
    params, spec, mc, _ = load_student(bc_path)
    runner = PlayRunner(config.seed, config.opponents)
    opt = optimizer(config)
    opt_state = opt.init(params)
    key = jax.random.key(config.seed + 4)

    @jax.jit
    def step(p, s, b):
        (loss, metrics), grad = jax.value_and_grad(
            lambda pp: ppo_loss(pp, b, config, mc), has_aux=True
        )(p)
        updates, s = opt.update(grad, s, p)
        return optax.apply_updates(p, updates), s, metrics

    total_deals, updates = 0, 0
    for iteration in range(iterations):
        batch, key, stats = collect_rollout(
            runner, params, spec, mc, key, config.rollout_steps, config
        )
        total_deals += stats["deals"]
        # Rollout data are discarded after these epochs, then refreshed.
        for _ in range(config.ppo_epochs):
            key, shuffle = jax.random.split(key)
            order = np.asarray(jax.random.permutation(shuffle, config.rollout_steps))
            for start in range(0, len(order), config.batch_size):
                ix = order[start : start + config.batch_size]
                mini = jax.tree_util.tree_map(lambda x: x[ix], batch)
                params, opt_state, metrics = step(params, opt_state, mini)
                updates += 1
        print(
            {
                "iteration": iteration + 1,
                "deals": total_deals,
                **{k: float(v) for k, v in metrics.items()},
            },
            flush=True,
        )
    save_arrays(
        output,
        pack_params(params),
        meta(
            "ppo",
            config,
            spec,
            iterations=iterations,
            model_config=asdict(mc),
            updates=updates,
            deals=total_deals,
            steps=iterations * config.rollout_steps,
        ),
    )
