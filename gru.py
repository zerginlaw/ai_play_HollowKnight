# import time
import time
import warnings

import newenv
import pyautogui
# import torch
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

pyautogui.PAUSE = 0.
from gymnasium.wrappers.frame_stack import FrameStack

from newenv import lock


class CustomCallback(BaseCallback):
    """
    在用ppo更新模型时，暂停游戏
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
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

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        更新policy前该方法被触发
        """
        with lock:
            newenv.rollouting = True
        env.allkeyup()

        pyautogui.press("esc")
        print("start")
        time.sleep(1)  # 为了修复bug
        monitor = (653, 216, 510, 346)  # {'height': 692, 'left': 143, 'top': 43, 'width': 1020}
        try:
            assert pyautogui.locateOnScreen(f'locator/esc.png',
                                            region=monitor,
                                            confidence=0.8) is not None
        except AssertionError:
            warnings.warn("do not find esc after 10s")

            pyautogui.press("esc")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        离开model.learn()之前被触发
        """
        pass


callback1 = CustomCallback()
callback2 = CheckpointCallback(
    save_path=r'stack4_new_models',  # 12和13对照
    save_freq=1000,
    name_prefix=r"209k_add_"
)
callback = CallbackList([callback1, callback2])
env = newenv.HKEnv()
num_stack = 4
env = FrameStack(env, num_stack)  # 传入6帧
if __name__ == '__main__':
    class CustomCNN(BaseFeaturesExtractor):

        def __init__(self,
                     observation_space: env.observation_space,
                     features_dim
                     # : int = 144 * 4
                     ):
            super(CustomCNN, self).__init__(observation_space, features_dim)

            n_input_channels = observation_space.shape[1] * num_stack
            self.cnn = nn.Sequential(
                # 4,160,160
                nn.BatchNorm2d(4),
                nn.Conv2d(n_input_channels, 12, kernel_size=3, stride=1, padding=0),  # 24,158,158
                nn.MaxPool2d(2),  # 6,79,79
                nn.LeakyReLU(),

                nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=0),  # 48,77,77
                nn.LeakyReLU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0),  # 96,75,75
                nn.MaxPool2d(3),  # 96,25,25
                nn.LeakyReLU(),

                nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=0),  # 192,23,23
                nn.LeakyReLU(),

                nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=0),  # 384,21,21
                nn.MaxPool2d(3),  # 384,7,7
                nn.LeakyReLU(),

                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),  # 768,5,5
                nn.LeakyReLU(),

                nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=0),  # 1536,3,3
                nn.LeakyReLU(),
                nn.Conv2d(768, 1536, kernel_size=3, stride=1, padding=0),  # 3072,1,1
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(1536, features_dim),
                nn.BatchNorm1d(features_dim),

                nn.LeakyReLU()
            )

            # with th.no_grad():
            #     n_flatten = self.cnn(
            #         th.as_tensor(observation_space.sample()[None]).float()
            #     ).shape[1]  # 3072
            #     # print(n_flatten)

        def forward(self, obs) -> th.Tensor:  # 4,3,160,160
            obs = obs.reshape(-1, 4, 160, 160)

            return self.cnn(obs)


    policy_kwargs = dict(
        activation_fn=th.nn.LeakyReLU,
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[128, 64, 64], vf=[256, 256, 256])
    )

    model = PPO("CnnPolicy",
                env=env,
                policy_kwargs=policy_kwargs, verbose=1, seed=1024,
                n_steps=100, batch_size=100,
                tensorboard_log=r"logs\gru_new2_nn",
                learning_rate=0.003, n_epochs=3,
                gamma=0.8, gae_lambda=0.85,
                clip_range=0.3,
                ent_coef=0.5, vf_coef=0.5
                )
    model = model.learn(total_timesteps=int(
        13333 * 12
        # 2000
    ),
        # progress_bar=True,
        callback=callback
    )

    # model.save(r"D:\HollowKnight_cpu\gru_new_nn_models\1")
    print(model.policy)
