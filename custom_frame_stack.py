import time

import gymnasium
import numpy as np
import pyautogui
from newenv import DashState
from newenv import INITIAL_ACTION


class CustomDictobs(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "obs": gymnasium.spaces.Box(low=0, high=255, shape=(4, 3, 81, 81), dtype=np.uint8),
                "able_a": gymnasium.spaces.MultiBinary(1),
                "dash_state": gymnasium.spaces.MultiBinary([1, 2]),
                "action_list": gymnasium.spaces.MultiBinary(20)  # 过去四帧的动作
            })

    def _dash(self):
        if self.unwrapped.dash_state == DashState.NODASH:
            dash = np.array([0, 0])
        elif self.unwrapped.dash_state == DashState.WHITEDASH:
            dash = np.array([0, 1])
        elif self.unwrapped.dash_state == DashState.BLACKDASH:
            dash = np.array([1, 0])
        return dash

    def observation(self, observation):
        """输入是(4,3, 81, 81)"""
        dash = self._dash().reshape((1, 2))
        action_list = np.array(self.unwrapped.info["action_list"])
        return {"obs": observation,
                "able_a": np.array(self.unwrapped.able_a).reshape((1, 1)),
                "dash_state": dash,
                "action_list": action_list
                }


class EpisodicLifeEnv(gymnasium.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gymnasium.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.lives()
        if lives is not None:
            if lives < self.lives and lives > 0:
                # for Qbert sometimes we stay in lives == 0 condition for a few frames
                # so it's important to keep lives > 0, so that we only reset once
                # the environment advertises done.
                print("lives", lives)
                truncated = True  #
            self.lives = lives
        # print(lives)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
            # self.lives = self.env.unwrapped.lives()
            return obs, info
        else:
            # no-op step to advance from terminal/lost life state
            if pyautogui.locateOnScreen(f'locator/esc.png',
                                        region=(653, 216, 510, 346),
                                        confidence=0.8):
                pyautogui.press("esc")
                time.sleep(1)

            obs, _, _, _, info = self.env.step(INITIAL_ACTION)
            # self.lives = self.env.unwrapped.lives()
            return obs, info
