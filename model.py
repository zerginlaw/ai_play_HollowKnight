# import time
import time
import warnings

import newenv
# import numpy
# import numpy as np
import pyautogui
# import torch
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

pyautogui.PAUSE = 0.
from gymnasium.wrappers.frame_stack import FrameStack
from custom_frame_stack import CustomDictobs, EpisodicLifeEnv
from newenv import lock


class CustomCallback(BaseCallback):
    """
    在用ppo更新模型时，暂停游戏
    lose或者win时记录boss血量
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.last_collecttime = None
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        第一个rollout开始之前这个方法被call
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        新的sample被收集时该方法被触发
        """
        # print("on_rollout_start")

    pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        这个方法被model call在每次进行env.step()

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        try:
            if ("boss_health" in self.model.env.envs[0].unwrapped.info
                    and self.model.env.envs[0].unwrapped.info["boss_health"] is not None):
                self.logger.record_mean("boss_health", self.model.env.envs[0].unwrapped.info["boss_health"])
        except Exception as e:
            warnings.warn(e)
        # AttributeError: 'DummyVecEnv' object has no attribute 'info'
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        更新policy前该方法被触发
        """
        with lock:
            newenv.rollouting = True
        env.allkeyup()
        now = time.time()
        if self.last_collecttime is not None:
            collect_time = now - self.last_collecttime
            print(f"收集时间为{divmod(collect_time, 60)[0]}分{divmod(collect_time, 60)[1]}秒")
        self.last_collecttime = now

        pyautogui.press("esc")
        print("start")
        time.sleep(1)  # 为了修复bug
        monitor = (653, 216, 510, 346)  # {'height': 692, 'left': 143, 'top': 43, 'width': 1020}
        try:
            assert pyautogui.locateOnScreen(f'locator/esc.png',
                                            region=monitor,
                                            confidence=0.8) is not None
        except AssertionError:
            warnings.warn("do not find esc after 1s")

            pyautogui.press("esc")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        离开model.learn()之前被触发
        """
        pass


callback1 = CustomCallback()
callback2 = CheckpointCallback(
    save_path=r'zote01models',
    save_freq=8193,
    name_prefix=r"0k_add_"
)
callback = CallbackList([callback1, callback2])
env = newenv.HKEnv()
num_stack = 4
env = FrameStack(env, num_stack)  # 传入4帧
env = EpisodicLifeEnv(env)

env = CustomDictobs(env)  # 加入是否可以攻击的字典

if __name__ == '__main__':
    class CustomCNN(BaseFeaturesExtractor):

        def __init__(self,
                     observation_space: env.observation_space,
                     features_dim
                     # : int = 144 * 4
                     ):
            super(CustomCNN, self).__init__(observation_space["obs"], features_dim)

            self.n_input_channels = observation_space["obs"].shape[1] * num_stack  # 12
            self.cnn1 = nn.Sequential(
                # 12,81,81
                nn.Conv2d(self.n_input_channels, 36, kernel_size=3, stride=3, padding=0),
                # 36,27,27
                nn.LeakyReLU(),
            )
            self.cnn2 = nn.Sequential(
                # 36,27,27
                nn.Conv2d(36, 72, kernel_size=3, stride=2, padding=0),
                # 72,13,13
                nn.LeakyReLU(),

            )
            self.cnn3 = nn.Sequential(
                # 72,13,13
                nn.Conv2d(72, 36, kernel_size=3, stride=2, padding=0),
                # 72,6,6
                nn.LeakyReLU(),

            )
            self.cnn4 = nn.Sequential(
                # 72,6,6
                nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0),
                # 36,4,4
                nn.LeakyReLU(),

            )
            self.cnn5 = nn.Sequential(
                # 36,4,4
                nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0),
                # 18,2,2
                nn.LeakyReLU(),

            )

            self.flatten = nn.Flatten()

            with th.no_grad():
                n_flatten = 0
                obs = th.as_tensor(observation_space["obs"].sample().reshape(-1, 12, 81, 81)).float()

                obs = self.cnn1(obs)

                obs = self.cnn2(obs)
                n_flatten += self.flatten(obs).shape[1]

                obs = self.cnn3(obs)
                n_flatten += self.flatten(obs).shape[1]

                obs = self.cnn4(obs)
                n_flatten += self.flatten(obs).shape[1]

                obs = self.cnn5(obs)
                n_flatten += self.flatten(obs).shape[1]

                print(n_flatten)  # 8112

            self.sequential = nn.Sequential(
                nn.Linear(n_flatten, features_dim - 3),
                nn.LeakyReLU()
            )

        def forward(self, obs) -> th.Tensor:  # 4,3,160,160
            a = self.cnn1(obs["obs"].reshape(-1, self.n_input_channels, 81, 81))
            b = self.cnn2(a)
            c = self.cnn3(b)
            d = self.cnn4(c)
            e = self.cnn5(d)

            able_a = obs["able_a"]
            dash = self.flatten(obs["dash_state"])

            result1 = th.cat((
                self.flatten(b),
                self.flatten(c),
                self.flatten(d),
                self.flatten(e),
            ), dim=-1)

            return th.cat((self.sequential(result1), able_a, dash), dim=-1)


    policy_kwargs = dict(
        activation_fn=th.nn.LeakyReLU,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=1536),
        net_arch=dict(pi=[512], vf=[512])  # 这里用列表指定网络结构，比如pi=[512,512]
    )

    model = PPO("MultiInputPolicy",
                env=env,
                policy_kwargs=policy_kwargs, verbose=1, seed=512,
                n_steps=16384, batch_size=512,
                # n_steps=5, batch_size=5,
                tensorboard_log=r"logs\zote01",
                learning_rate=0.0006, n_epochs=20,
                gamma=0.9, gae_lambda=0.85,
                clip_range=0.2,
                ent_coef=0.5, vf_coef=0.7
                )
    model = model.learn(total_timesteps=int(
        16384 * 5
        # 2000
    ),
        progress_bar=True,
        callback=callback
    )

    # model.save(r"D:\HollowKnight_cpu\gru_new_nn_models\1")
    print(model.policy)
