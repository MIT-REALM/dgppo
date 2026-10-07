import jax.numpy as jnp
import jax.random as jr
import jax

from typing import Optional

from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action, Array, Pos2d, Reward
from dgppo.env.mpe.base import MPEEnvState, MPEEnvGraphsTuple
from dgppo.env.utils import get_node_goal_rng
from dgppo.env.mpe.mpe_spread import MPESpread


class MPEFormation(MPESpread):

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "formation_radius": 0.5,
        "n_obs": 3,
        "obs_radius": 0.05,
        "default_area_size": 1.5,
        "dist2goal": 0.01
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = MPEFormation.PARAMS["default_area_size"] if area_size is None else area_size
        super(MPEFormation, self).__init__(num_agents, area_size, max_step, dt, params)
        self.num_goals = 1
        # Full observation changes communication range, not the target geometry.
        self.formation_radius = self.params.get("formation_radius", self.PARAMS["formation_radius"])
        if not 0 < self.formation_radius <= self.area_size / 2 - 2 * self.params["car_radius"]:
            raise ValueError("formation_radius must be positive and fit inside the arena with agent clearance")

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate agent
        states, _ = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None
        )

        # generate a landmark
        R = self.formation_radius
        landmark_key, key = jr.split(key)
        landmark = jr.uniform(landmark_key, (1, 2),
                              minval=R + 2 * self.params['car_radius'],
                              maxval=self.area_size - R - 2 * self.params['car_radius'])
        goals = self.landmark2goal(landmark, R)

        # randomly generate obstacles
        def get_obs(inp):
            this_key, _ = inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,),
                                        minval=self.params['car_radius'] * 3,
                                        maxval=self.area_size - self.params['car_radius'] * 3)

        def non_valid_obs(inp):
            _, this_obs = inp
            dist_min_agents = jnp.linalg.norm(states - this_obs, axis=1).min()
            dist_min_goals = jnp.linalg.norm(goals - this_obs, axis=1).min()
            collide_agent = dist_min_agents <= self.params["car_radius"] + self.params["obs_radius"]
            collide_goal = dist_min_goals <= self.params["car_radius"] * 2 + self.params["obs_radius"]
            out_region = (jnp.any(this_obs < self.params["car_radius"] * 3) |
                          jnp.any(this_obs > self.area_size - self.params["car_radius"] * 3))
            return collide_agent | collide_goal | out_region

        def get_valid_obs(carry, inp):
            this_key = inp
            use_key, this_key = jr.split(this_key, 2)
            obs_candidate = jr.uniform(use_key, (2,), minval=0, maxval=self.area_size)
            _, valid_obs = jax.lax.while_loop(non_valid_obs, get_obs, (this_key, obs_candidate))
            return carry, valid_obs

        obs_keys = jr.split(key, self.params["n_obs"])
        _, obs = jax.lax.scan(get_valid_obs, None, obs_keys)

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        landmark = jnp.concatenate([landmark, jnp.zeros_like(landmark)], axis=1)
        obs = jnp.concatenate([obs, jnp.zeros_like(obs)], axis=1)

        env_state = MPEEnvState(states, landmark, obs)

        return self.get_graph(env_state)

    def landmark2goal(self, landmarks: Pos2d, R: float) -> Pos2d:
        assert landmarks.shape == (1, 2)
        thetas = jnp.linspace(0, 2 * jnp.pi, self.num_agents + 1)[:-1]
        goals = landmarks + R * jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
        return goals

    def get_reward(self, graph: MPEEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        landmark = graph.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
        goals = self.landmark2goal(landmark, self.formation_radius)

        # each goal finds the nearest agent
        reward = jnp.zeros(()).astype(jnp.float32)
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        reward -= dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward
