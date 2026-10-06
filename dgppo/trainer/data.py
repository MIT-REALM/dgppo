from typing import NamedTuple, Optional

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    graph: GraphsTuple
    actions: Action
    rnn_states: Array  # Actor carry before processing the corresponding graph.
    rewards: Reward
    costs: Cost
    dones: Done
    log_pis: Optional[Array]
    next_graph: GraphsTuple

    @property
    def length(self) -> int:
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        return self.length * self.time_horizon
