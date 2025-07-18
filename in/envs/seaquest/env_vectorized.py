from typing import Sequence
from env_common.env_vectorized import VectorizedNudgeBaseEnv
from env_common.env_utils import make_env
import torch as th
from ocatari.ram.seaquest import MAX_NB_OBJECTS
import gymnasium as gym
from hackatari.core import HackAtari


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "seaquest"
    pred2action = {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str,
        n_envs: int,
        render_mode="rgb_array",
        render_oc_overlay=False,
        seed=None,
    ):
        super().__init__(mode)
        # set up multiple envs
        self.n_envs = n_envs
        self.envs = [
            HackAtari(
                env_name="ALE/Seaquest-v5",
                mode="ram",
                obs_mode="ori",
                rewardfunc_path="in/envs/seaquest/blenderl_reward.py",
                render_mode=render_mode,
                render_oc_overlay=render_oc_overlay,
            )
            for i in range(n_envs)
        ]
        # apply wrapper to _env in OCAtari
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)

        # for learning script from cleanrl
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 43
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        logic_states = []
        neural_states = []
        seed_i = self.seed
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            # lazy frame to tensor
            obs = th.tensor(obs).float()
            state = env.objects
            raw_state = obs  # self.env.dqn_obs
            logic_state, neural_state = self.extract_logic_state(
                state
            ), self.extract_neural_state(raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        return th.stack(logic_states), th.stack(neural_states)

    def step(self, actions, is_mapped: bool = False):
        assert (
            len(actions) == self.n_envs
        ), "Invalid number of actions: n_actions is {} and n_envs is {}".format(
            len(actions), self.n_envs
        )
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            # make a step in the env
            obs, reward, truncation, done, info = env.step(action)
            # lazy frame to tensor
            obs = th.tensor(obs).float()
            # get logic and neural state
            state = env.objects
            raw_state = obs
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
            # store final info

        # observations = th.stack(observations)
        return (
            (th.stack(logic_states), th.stack(neural_states)),
            rewards,
            truncations,
            dones,
            infos,
        )

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "OxygenBar":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = (
                    obj.orientation.value if obj.orientation is not None else 0
                )
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
        return state

    def extract_neural_state(self, raw_input_state):
        return raw_input_state

    def close(self):
        for env in self.envs:
            env.close()
