import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import os

from typing import Callable, TYPE_CHECKING
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey, Array
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, LogPi, RNN_States]
    init_rnn_state: Array
    key: PRNGKey

    Returns
    -------
    data: Rollout
    """
    key_x0, key_z0, key = jax.random.split(key, 3)
    init_graph = env.reset(key_x0)

    def body(data, key_):
        graph, rnn_state = data
        action, log_pi, new_rnn_state = actor(graph, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)

        return ((next_graph, new_rnn_state),
                (graph, action, rnn_state, reward, cost, done, log_pi, next_graph))

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
        jax.lax.scan(body, (init_graph, init_rnn_state), keys, length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs)
    return rollout_data


def test_rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False
):
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body_(data, key_):
        graph, rnn_state = data
        if not stochastic:
            action, new_rnn_state = actor(graph, rnn_state)
        else:
            action, new_rnn_state = actor(graph, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return (next_graph, new_rnn_state), (graph, action, rnn_state, reward, cost, done, None, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
        jax.lax.scan(body_,
                     (init_graph, init_rnn_state),
                     keys,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs)
    return rollout_data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected():
    return internet()


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)


def plot_rnn_states(rnn_states: Array, name: str, path: str):
    """
    rnn_states: (T, n_layer, n_agent, n_carry, hid_size)
    """
    T, n_layer, n_agent, n_carry, hid_size = rnn_states.shape
    for i_layer in range(n_layer):
        fig, ax = plt.subplots(nrows=n_agent, ncols=n_carry, figsize=(10, 20))
        for i_agent in range(n_agent):
            for i_carry in range(n_carry):
                ax[i_agent, i_carry].plot(rnn_states[:, i_layer, i_agent, i_carry, :])
                ax[i_agent, i_carry].set_title(f'Agent {i_agent}, carry {i_carry}, layer {i_layer}')
                ax[i_agent, i_carry].set_xlabel('Time step')
                ax[i_agent, i_carry].set_ylabel('State value')
        fig.tight_layout()
        plt.savefig(os.path.join(path, f'rnn_states_{name}_layer{i_layer}.png'))
