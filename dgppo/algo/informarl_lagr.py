import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle
import flax.linen as nn

from typing import Tuple
from flax.training.train_state import TrainState

from ..utils.typing import Params, Array
from ..utils.utils import jax_vmap, tree_index
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..env.base import MultiAgentEnv
from ..algo.module.value import ValueNet
from .utils import compute_dec_ocp_gae
from .informarl import InforMARL


class InforMARLLagr(InforMARL):

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
            lagr_init: float = 0.78,
            lr_lagr: float = 1e-7,
            **kwargs
    ):
        super(InforMARLLagr, self).__init__(
            env, node_dim, edge_dim, state_dim, action_dim, n_agents, 0., actor_gnn_layers, Vl_gnn_layers,
            gamma, lr_actor, lr_Vl, batch_size, epoch_ppo, clip_eps, gae_lambda, coef_ent, max_grad_norm, seed,
            use_rnn, rnn_layers, rnn_step, use_lstm
        )

        # set hyperparameters
        self.lr_Vh = lr_Vh
        self.Vh_gnn_layers = Vh_gnn_layers
        self.lagr_init = lagr_init
        self.lr_lagr = lr_lagr

        # cost value function
        self.Vh = ValueNet(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            n_out=env.n_cost,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=Vh_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            decompose=True,
            use_global_info=True
        )

        # initialize the rnn state
        rnn_state_key, self.key = jr.split(self.key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_Vh_rnn_state = jax_vmap(self.Vh.initialize_carry)(rnn_state_key)  # (n_agents, rnn_state_dim)
        if type(init_Vh_rnn_state) is tuple:
            init_Vh_rnn_state = jnp.stack(init_Vh_rnn_state, axis=1)  # (n_agents, n_carries, rnn_state_dim)
        else:
            init_Vh_rnn_state = jnp.expand_dims(init_Vh_rnn_state, axis=1)
        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_Vh_rnn_state = init_Vh_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        Vh_key, self.key = jr.split(self.key)
        Vh_params = self.Vh.net.init(Vh_key, self.nominal_graph, self.init_Vh_rnn_state, self.n_agents)
        Vh_optim = optax.adam(learning_rate=lr_Vh)
        self.Vh_optim = optax.apply_if_finite(Vh_optim, 1_000_000)
        self.Vh_train_state = TrainState.create(
            apply_fn=self.Vh.get_value,
            params=Vh_params,
            tx=self.Vh_optim
        )

        # initialize the lagrange multiplier
        self.ah_lagr = jnp.ones((self.n_agents, self._env.n_cost)) * self.lagr_init

    @property
    def config(self) -> dict:
        return super().config | {
            "lr_Vh": self.lr_Vh,
            "Vh_gnn_layers": self.Vh_gnn_layers,
            "lagr_init": self.lagr_init,
            "lr_lagr": self.lr_lagr
        }

    @property
    def params(self) -> Params:
        return {
            "policy": self.policy_train_state.params,
            "Vl": self.Vl_train_state.params,
            "Vh": self.Vh_train_state.params
        }

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
            Vl_train_state, Vh_train_state, policy_train_state, ah_lagr, update_info = self.update_inner(
                self.Vl_train_state,
                self.Vh_train_state,
                self.policy_train_state,
                self.ah_lagr,
                rollout,
                batch_idx,
                rnn_chunk_ids
            )
            self.Vl_train_state = Vl_train_state
            self.Vh_train_state = Vh_train_state
            self.policy_train_state = policy_train_state
            self.ah_lagr = ah_lagr
        return update_info

    def scan_Vh(
            self, rollout: Rollout, init_rnn_state: Array, Vh_params: Params
    ) -> Tuple[Array, Array, Array]:
        T_graphs = rollout.graph

        def body_(rnn_state, graph):
            Vh, new_rnn_state = self.Vh.get_value(Vh_params, graph, rnn_state)
            return new_rnn_state, (Vh, rnn_state)

        final_rnn_state, (Tah_Vh, T_rnn_states) = jax.lax.scan(body_, init_rnn_state, T_graphs)

        return Tah_Vh, T_rnn_states, final_rnn_state

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_inner(
            self,
            Vl_train_state: TrainState,
            Vh_train_state: TrainState,
            policy_train_state: TrainState,
            ah_lagr: Array,
            rollout: Rollout,
            batch_idx: Array,
            rnn_chunk_ids: Array,
    ) -> Tuple[TrainState, TrainState, TrainState, Array, dict]:
        # rollout: (n_env, T, n_agent, ...)
        b, T, a, _ = rollout.actions.shape

        # calculate Vl
        bT_Vl, bT_Vl_rnn_states, final_Vl_rnn_states = jax.vmap(
            ft.partial(self.scan_Vl,
                       init_Vl_rnn_state=self.init_Vl_rnn_state,
                       Vl_params=Vl_train_state.params)
        )(rollout)

        def final_Vl_fn_(graph, rnn_state_Vl):
            Vl, _ = self.Vl.get_value(Vl_train_state.params, tree_index(graph, -1), rnn_state_Vl)
            return Vl.squeeze(0).squeeze(0)

        b_final_Vl = jax.vmap(final_Vl_fn_)(rollout.next_graph, final_Vl_rnn_states)
        bTp1_Vl = jnp.concatenate([bT_Vl, b_final_Vl[:, None]], axis=1)
        assert bTp1_Vl.shape[:2] == (b, T + 1)

        # calculate Vh
        bTah_Vh, bT_Vh_rnn_states, final_Vh_rnn_states = jax.vmap(
            ft.partial(self.scan_Vh,
                       init_rnn_state=self.init_Vh_rnn_state,
                       Vh_params=Vh_train_state.params)
        )(rollout)

        def final_Vh_fn_(graph, rnn_state_Vh):
            Vh, _ = self.Vh.get_value(Vh_train_state.params, tree_index(graph, -1), rnn_state_Vh)
            return Vh

        bah_final_Vh = jax.vmap(final_Vh_fn_)(rollout.next_graph, final_Vh_rnn_states)
        bTp1ah_Vh = jnp.concatenate([bTah_Vh, bah_final_Vh[:, None]], axis=1)
        assert bTp1ah_Vh.shape[:4] == (b, T + 1, self.n_agents, self._env.n_cost)

        # calculate Dec-OCP GAE
        bTah_Qh, bT_Ql = jax.vmap(
            ft.partial(compute_dec_ocp_gae, disc_gamma=self.gamma, gae_lambda=self.gae_lambda)
        )(Tah_hs=jnp.clip(rollout.costs, a_min=0),
          T_l=-rollout.rewards,
          Tp1ah_Vh=bTp1ah_Vh,
          Tp1_Vl=bTp1_Vl)

        # calculate advantages and normalize
        # cost advantage
        bT_Al: Array = bT_Ql - bT_Vl
        bT_Al = (bT_Al - bT_Al.mean(axis=1, keepdims=True)) / (bT_Al.std(axis=1, keepdims=True) + 1e-8)
        bTa_Al = bT_Al[:, :, None].repeat(self.n_agents, axis=-1)
        bTa_Al = -bTa_Al
        assert bTa_Al.shape == (b, T, self.n_agents)

        # constraint advantage
        bTah_Ah: Array = bTah_Qh - bTah_Vh
        bTah_Ah = (bTah_Ah - bTah_Ah.mean(axis=1, keepdims=True)) / (bTah_Ah.std(axis=1, keepdims=True) + 1e-8)
        assert bTah_Ah.shape == (b, T, self.n_agents, self._env.n_cost)

        # total advantage
        bTa_A = bTa_Al - (bTah_Ah * ah_lagr[None, None]).mean(axis=-1)
        assert bTa_A.shape == (b, T, self.n_agents)

        # ppo update
        def update_fn(carry, idx):
            Vl_model, Vh_model, policy_model, lagr_lambda = carry
            rollout_batch = jtu.tree_map(lambda x: x[idx], rollout)
            Vl_model, Vl_info = self.update_Vl(
                Vl_model, rollout_batch, bT_Ql[idx], bT_Vl_rnn_states[idx], rnn_chunk_ids)
            Vh_model, Vh_info = self.update_Vh(
                Vh_model, rollout_batch, bTah_Qh[idx], bT_Vh_rnn_states[idx], rnn_chunk_ids)
            policy_model, policy_info = self.update_policy(policy_model, rollout_batch, bTa_A[idx], rnn_chunk_ids)
            lagr_lambda, lagr_info = self.update_lagr(
                lagr_lambda, policy_model, rollout_batch, bTah_Vh[idx], bTah_Ah[idx])

            return (Vl_model, Vh_model, policy_model, lagr_lambda), (Vl_info | Vh_info | policy_info | lagr_info)

        (Vl_train_state, Vh_train_state, policy_train_state, ah_lagr), update_info = jax.lax.scan(
            update_fn, (Vl_train_state, Vh_train_state, policy_train_state, ah_lagr), batch_idx)

        # get training info of the last PPO epoch
        info = jtu.tree_map(lambda x: x[-1], update_info)

        return Vl_train_state, Vh_train_state, policy_train_state, ah_lagr, info

    def update_Vh(
            self,
            Vh_train_state: TrainState,
            rollout: Rollout,
            bTah_Qh: Array,
            bT_rnn_states: Array,
            rnn_chunk_ids: Array
    ) -> Tuple[TrainState, dict]:
        bcT_rollout = jtu.tree_map(lambda x: x[:, rnn_chunk_ids], rollout)
        bcTah_Qh = bTah_Qh[:, rnn_chunk_ids]
        bc_rnn_state_inits = jax.lax.stop_gradient(bT_rnn_states[:, rnn_chunk_ids[:, 0]])

        def get_loss_(params):
            bcTah_Vh, bcT_Vh_rnn_states, final_Vh_rnn_states = jax.vmap(jax.vmap(
                ft.partial(self.scan_Vh,
                           Vh_params=params)
            ))(bcT_rollout, bc_rnn_state_inits)

            loss_Vh = optax.l2_loss(bcTah_Vh, bcTah_Qh).mean()
            return loss_Vh

        loss, grad = jax.value_and_grad(get_loss_)(Vh_train_state.params)
        critic_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        Vh_train_state = Vh_train_state.apply_gradients(grads=grad)
        return Vh_train_state, {'Vh/loss': loss,
                                'Vh/grad_norm': grad_norm,
                                'Vh/has_nan': critic_has_nan,
                                'Vh/max_target': jnp.max(bcTah_Qh),
                                'Vh/min_target': jnp.min(bcTah_Qh)}

    def update_lagr(
            self,
            ah_lagr_lambda: Array,
            policy_train_state: TrainState,
            rollout: Rollout,
            bTah_Vh: Array,
            bTah_Ah: Array
    ) -> Tuple[Array, dict]:
        b_rnn_state_inits = jax.lax.stop_gradient(rollout.rnn_states[:, 0])

        action_key = jr.fold_in(self.key, policy_train_state.step)
        bT_action_keys = jr.split(action_key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[:2] + (2,))

        # calculate log_pi
        bTa_log_pis, bTa_policy_entropy, bT_rnn_states, final_rnn_states = jax.vmap(
            ft.partial(self.scan_eval_action,
                       actor_params=policy_train_state.params)
        )(rollout.graph, rollout.actions, b_rnn_state_inits, bT_action_keys)

        bTa_ratio = jnp.exp(bTa_log_pis - rollout.log_pis)
        ah_delta_lagr = -(bTah_Vh * (1 - self.gamma) + bTa_ratio[:, :, :, None] * bTah_Ah).mean(axis=(0, 1))
        ah_lagr_lambda = nn.relu(ah_lagr_lambda - ah_delta_lagr * self.lr_lagr)
        return ah_lagr_lambda, {'policy/lagr_mean': ah_lagr_lambda.mean()}

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.policy_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.Vl_train_state.params, open(os.path.join(model_dir, 'Vl.pkl'), 'wb'))
        pickle.dump(self.Vh_train_state.params, open(os.path.join(model_dir, 'Vh.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))

        self.policy_train_state = \
            self.policy_train_state.replace(params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb')))
        self.Vl_train_state = \
            self.Vl_train_state.replace(params=pickle.load(open(os.path.join(path, 'Vl.pkl'), 'rb')))
        self.Vh_train_state = \
            self.Vh_train_state.replace(params=pickle.load(open(os.path.join(path, 'Vh.pkl'), 'rb')))
