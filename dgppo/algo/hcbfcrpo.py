import jax.numpy as jnp
import jax.random as jr
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax

from ..utils.typing import Params, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import jax_vmap, tree_index
from .utils import compute_dec_ocp_gae
from ..trainer.data import Rollout
from ..env.base import MultiAgentEnv
from .dgppo import DGPPO


class HCBFCRPO(DGPPO):
    """DGPPO with a hand-crafted CBF."""

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            actor_gnn_layers: int = 2,
            Vl_gnn_layers: int = 2,
            Vh_gnn_layers: int = 1,
            gamma: float = 0.99,
            lr_actor: float = 3e-4,
            lr_Vl: float = 1e-3,
            lr_Vh: float = 1e-3,
            batch_size: int = 8192,
            epoch_ppo: int = 1,
            clip_eps: float = 0.25,
            gae_lambda: float = 0.95,
            coef_ent: float = 1e-2,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            rnn_step: int = 16,
            use_lstm: bool = False,
            alpha: float = 10.0,
            cbf_eps: float = 1e-2,
            cbf_weight: float = 1.0,
            train_steps: int = 1e5,
            cbf_schedule: bool = True,
            **kwargs
    ):
        super(HCBFCRPO, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            actor_gnn_layers=actor_gnn_layers,
            Vl_gnn_layers=Vl_gnn_layers,
            Vh_gnn_layers=Vh_gnn_layers,
            gamma=gamma,
            lr_actor=lr_actor,
            lr_Vl=lr_Vl,
            lr_Vh=lr_Vh,
            batch_size=batch_size,
            epoch_ppo=epoch_ppo,
            clip_eps=clip_eps,
            gae_lambda=gae_lambda,
            coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,
            seed=seed,
            use_rnn=use_rnn,
            rnn_layers=rnn_layers,
            rnn_step=rnn_step,
            use_lstm=use_lstm,
            alpha=alpha,
            cbf_eps=cbf_eps,
            cbf_weight=cbf_weight,
            train_steps=train_steps,
            cbf_schedule=cbf_schedule,
            **kwargs
        )

    def get_Vh(
            self, graph: GraphsTuple, rnn_state: Array, params: Optional[Params] = None
    ) -> Array:
        return self._env.get_cost(graph)

    def update(self, rollout: Rollout, step: int) -> dict:
        key, self.key = jr.split(self.key)

        update_info = {}
        assert rollout.dones.shape[0] * rollout.dones.shape[1] >= self.batch_size
        for i_epoch in range(self.epoch_ppo):
            idx = np.arange(rollout.dones.shape[0])
            np.random.shuffle(idx)
            rnn_chunk_ids = jnp.arange(rollout.dones.shape[1])
            rnn_chunk_ids = jnp.array(jnp.array_split(rnn_chunk_ids, rollout.dones.shape[1] // self.rnn_step))
            batch_idx = jnp.array(jnp.array_split(idx, idx.shape[0] // (self.batch_size // rollout.dones.shape[1])))
            Vl_train_state, policy_train_state, update_info = self.update_inner(
                self.Vl_train_state,
                self.policy_train_state,
                rollout,
                batch_idx,
                rnn_chunk_ids,
                jnp.array(step)
            )
            self.Vl_train_state = Vl_train_state
            self.policy_train_state = policy_train_state
        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            Vl_train_state: TrainState,
            policy_train_state: TrainState,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
            step: Array
    ) -> Tuple[TrainState, TrainState, dict]:
        # rollout: (b, T, a, ...)
        b, T, a, _ = rollout.actions.shape

        # calculate Vl
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(self.scan_Vl,
                       init_Vl_rnn_state=self.init_Vl_rnn_state,
                       Vl_params=Vl_train_state.params)
        )(rollout)

        def final_Vl_fn_(graph, rnn_state):
            Vl, _ = self.Vl.get_value(Vl_train_state.params, tree_index(graph, -1), rnn_state)
            return Vl.squeeze(0).squeeze(0)

        b_final_Vl = jax_vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape[:2] == (b, T + 1)

        # calculate Vh
        bTah_Vh = jax.vmap(jax.vmap(ft.partial(
            self.get_Vh, params={'Vh': None})))(rollout.graph, rollout.rnn_states)

        final_Vh_fn_ = ft.partial(self.get_final_Vh, params={'policy': policy_train_state.params, 'Vh': None})
        final_Vh = jax.vmap(final_Vh_fn_)(rollout)

        bTp1ah_Vh = jnp.concatenate([bTah_Vh, final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape[:4] == (b, T + 1, a, self._env.n_cost)

        # calculate Dec-EFOCP GAE
        bTah_Qh, bT_Ql = jax.vmap(
            ft.partial(compute_dec_ocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=rollout.costs,
          T_l=-rollout.rewards,
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        # cost advantage
        bT_Al = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)

        # CBF advantage
        bTah_cbf_deriv = (bTp1ah_Vh[:, 1:] - bTah_Vh) / self._env.dt + self.alpha * bTah_Vh
        bTah_Acbf = jnp.maximum(bTah_cbf_deriv + self.cbf_eps, 0)

        # merge advantage
        bTa_is_safe = (bTah_cbf_deriv <= 0).min(axis=-1)
        safe_data = bTa_is_safe.mean()
        bTa_A = jnp.where(bTa_is_safe, bTa_Al, jnp.zeros_like(bTa_Al))
        if self.cbf_schedule:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_schedule_fn(step)
        else:
            bTa_A += bTah_Acbf.max(axis=-1) * self.cbf_weight

        # reverse advantage
        bTa_A = -bTa_A

        # ppo update
        def update_fn(carry, idx):
            Vl_model, policy_model = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_Ql[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy(policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            return (Vl_model, policy_model), (Vl_info | policy_info)

        (Vl_train_state, policy_train_state), info = lax.scan(
            update_fn, (Vl_train_state, policy_train_state), batch_idx
        )

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], info) | {'eval/safe_data': safe_data}

        return Vl_train_state, policy_train_state, info
