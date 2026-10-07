from copy import deepcopy
from typing import Optional

from .base import MultiAgentEnv
from dgppo.env.mpe import MPETarget, MPESpread, MPELine, MPEFormation, MPECorridor, MPEConnectSpread
from dgppo.env.lidar_env import LidarSpread, LidarTarget, LidarLine, LidarBicycleTarget
from dgppo.env.vmas import VMASWheel, VMASReverseTransport


ENV = {

    'MPETarget': MPETarget,
    'MPESpread': MPESpread,
    'MPELine': MPELine,
    'MPEFormation': MPEFormation,
    'MPECorridor': MPECorridor,
    'MPEConnectSpread': MPEConnectSpread,
    'LidarSpread': LidarSpread,
    'LidarTarget': LidarTarget,
    'LidarLine': LidarLine,
    'LidarBicycleTarget': LidarBicycleTarget,
    'VMASReverseTransport': VMASReverseTransport,
    'VMASWheel': VMASWheel,
}


DEFAULT_MAX_STEP = 128


def make_env(
        env_id: str,
        num_agents: int,
        max_step: int = None,
        full_observation: bool = False,
        num_obs: Optional[int] = None,
        n_rays: Optional[int] = None,
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    # Overrides belong to this environment, not the shared class defaults.
    params = deepcopy(ENV[env_id].PARAMS)
    max_step = DEFAULT_MAX_STEP if max_step is None else max_step
    if num_obs is not None:
        params['n_obs'] = num_obs
    if n_rays is not None:
        params['n_rays'] = n_rays
    if full_observation:
        area_size = params['default_area_size']
        params['comm_radius'] = area_size * 10
    return ENV[env_id](
        num_agents=num_agents,
        area_size=None,
        max_step=max_step,
        dt=0.03,
        params=params
    )
