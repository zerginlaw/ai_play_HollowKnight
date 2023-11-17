import enum
import time
import warnings

import gymnasium
import numpy as np

import pyautogui
from mss.windows import MSS
import cv2

import gc
import threading

# pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.
rollouting = False
lock = threading.Lock()


#
# def busy_sleep(seconds_to_sleep):
#     start = time.perf_counter()
#     while time.perf_counter() < start + seconds_to_sleep:
#         pass


class Actions(enum.Enum):
    pass


class Displacement(Actions):
    NOTJUMP = 0  # try to not jump
    JUMP = 1  # try to jump in next 0.37s
    # DASH = 2


class Move(Actions):
    STAY = 0
    LEFT = 1
    RIGHT = 2


class Attack(Actions):
    # NOTATTACK = 0
    ATTACK = 1
    DONWATTACK = 2
    # SPELL = 2


class Is_attack_then_move(Actions):
    no = 0
    yes = 1


class HKEnv(gymnasium.Env):
    metadata = {"render_modes": []}
    render_mode = ""
    reward_range = (-float("inf"), float("inf"))

    DISPLACEMENT_KEYMAP = {1: "K"}
    MOVE_KEYMAP = {1: "a", 2: "d"}
    ATTACK_KEYMAP = {1: "j"}
    DONWATTACK_KEYMAP = {2: "s"}

    HP_CKPT = np.array([52, 91, 129, 169, 207, 246, 286, 324, 363], dtype=int)

    def __init__(self, rgb=False, w1=0.6, w2=0.4, w3=-0.0001):
        self.w1 = w1  # 被打
        self.w2 = w2  # 击中
        self.w3 = w3  # ignore
        self.rgb = rgb
        # spec: EnvSpec | None = None #An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [len(Displacement), len(Move), len(Attack), len(Is_attack_then_move)])
        if self.rgb:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 160, 160), dtype=np.uint8)
        else:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(1, 160, 160), dtype=np.uint8)

        self.monitor = self._find_window()

        self._prev_time = None
        self._last_actions = [0, 0, 0]
        self._last_5jump = [0, 0, 0, 0, 0]

        self.prev_enemy_hp = None
        self.gametime = 1  # 用于转换boss时计数

        # _np_random: np.random.Generator | None = None

    def _checkifthinking(self):

        global rollouting

        with lock:
            rollouting = False

        monitor = self.monitor
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'] + monitor['height'] // 4,
                   monitor['width'] // 2,
                   monitor['height'] // 2)

        assert pyautogui.locateOnScreen(f'locator/esc.png',
                                        region=monitor,
                                        confidence=0.8) is not None
        print("end")
        pyautogui.press("esc")

    def step(self, actions, no_notattck=True, steptime=0.3):  # [len(Displacement), len(Move), len(Attack)]
        try:
            if rollouting:
                self._checkifthinking()
                time.sleep(0.5)
        except AssertionError:
            warnings.warn("do not find esc")
            self.close()
            return self._observe()[0], 0, True, False, {}
        if no_notattck:
            actions[2] += 1  # 取消不攻击

        self._take_action(actions)  # it takes time to ensure 0.3s existed before last time taking actions

        t = steptime - (time.time() - self._prev_time)  # Attack interval
        # print(f"t should be +，remaining:{t}")
        if t > 0:
            time.sleep(t)
            # busy_sleep(t)

        else:
            print(f"delayed:{-t}")

        self._prev_time = time.time()

        # busy_sleep(0.006)  # it takes time to see how much the hp changes,then returen rew,done,obs
        reward, done, obs = self._get_this_result(actions)

        if no_notattck:
            actions[2] -= 1  # 取消不攻击
        if done:
            self.close()

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None, changeboss=False):
        super().reset(seed=seed)

        if pyautogui.locateOnScreen(f'locator/esc.png',
                                    region=(653, 216, 510, 346),
                                    confidence=0.8):
            pyautogui.press("esc")
            time.sleep(1)

        while True:
            if self._find_menu():
                # print("find menu")
                break
            pyautogui.press('w')
            time.sleep(1)
        if not changeboss or not (self.gametime % 4 == 0 or self.gametime % 4 == 1) or self.gametime == 1:
            pyautogui.press('k')  # 直接进入

        else:
            pyautogui.press('q')
            time.sleep(1)
            pyautogui.press('d')
            pyautogui.press('j')
            time.sleep(2)
            while True:
                if self._find_menu():
                    # print("find menu")
                    break
                pyautogui.press('w')
                time.sleep(1)
            pyautogui.press('k')
        self.gametime += 1

        # wait for loading screen
        ready = False
        while True:
            obs = self._observe(force_gray=True)[0]
            is_loading = (obs < 20).sum() < 10
            if ready and not is_loading:
                break
            else:
                ready = is_loading
            time.sleep(0.1)
        time.sleep(2)

        self.prev_knight_hp, self.prev_enemy_hp = len(self.HP_CKPT), 1.
        self._prev_time = time.time()
        observe = self._observe()[0]
        return observe, None

    def allkeyup(self):
        for keymap in [self.DISPLACEMENT_KEYMAP, self.MOVE_KEYMAP, self.ATTACK_KEYMAP, self.DONWATTACK_KEYMAP]:
            for key in keymap.values():
                pyautogui.keyUp(key)

    def close(self):
        """
        do any necessary cleanup on the interaction
        should only be called before or after an episode
        """
        self.allkeyup()

        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self._last_actions = [0, 0, 0]
        self._last_5jump = [0, 0, 0, 0, 0]

        self._prev_time = None
        gc.collect()

    def _take_action(self, actions):
        """actions:a list like[1,2,2,1]"""  # jump then move, then attack

        if actions[0] == self._last_actions[0]:  # jump
            pass
        else:  # 01 or 10
            if actions[0] == 1:
                key = self.DISPLACEMENT_KEYMAP[actions[0]]
                pyautogui.keyDown(key)
            else:
                key = self.DISPLACEMENT_KEYMAP[1]
                pyautogui.keyUp(key)
        if actions[3] == 0:  # move_then_attack
            if actions[1] == self._last_actions[1]:  # move
                pass
            elif self._last_actions[1] == 0:  # 01,02
                key = self.MOVE_KEYMAP[actions[1]]
                pyautogui.keyDown(key)
            else:  # 10,12,20,21
                key = self.MOVE_KEYMAP[self._last_actions[1]]
                pyautogui.keyUp(key)
                if actions[1] != 0:  # 12,21
                    key = self.MOVE_KEYMAP[actions[1]]
                    pyautogui.keyDown(key)

            if actions[2] == 1 or actions[2] == 2:  # attack
                key = self.ATTACK_KEYMAP[1]
                if actions[2] == 2:
                    pyautogui.keyDown(self.DONWATTACK_KEYMAP[actions[2]])
                pyautogui.press(key)
                pyautogui.keyUp(self.DONWATTACK_KEYMAP[2])
        else:  # attack_then_move
            if actions[2] == 1 or actions[2] == 2:  # attack
                key = self.ATTACK_KEYMAP[1]
                if actions[2] == 2:
                    pyautogui.keyDown(self.DONWATTACK_KEYMAP[actions[2]])
                pyautogui.press(key)
                pyautogui.keyUp(self.DONWATTACK_KEYMAP[2])

            if actions[1] == self._last_actions[1]:  # move
                pass
            elif self._last_actions[1] == 0:  # 01,02
                key = self.MOVE_KEYMAP[actions[1]]
                pyautogui.keyDown(key)
            else:  # 10,12,20,21
                key = self.MOVE_KEYMAP[self._last_actions[1]]
                pyautogui.keyUp(key)
                if actions[1] != 0:  # 12,21
                    key = self.MOVE_KEYMAP[actions[1]]
                    pyautogui.keyDown(key)

        self._last_actions = actions
        del self._last_5jump[4]
        self._last_5jump.insert(0, actions[0])

    def _get_this_result(self, actions):
        obs, knight_hp, enemy_hp = self._observe()  # knight HP ?,  enemy HP:0->1

        done, win = self._check_done(knight_hp, enemy_hp)
        reward = self._get_reward(knight_hp, enemy_hp, win)
        reward += self._searchinmap(actions)
        reward -= 0.001  # 时间惩罚
        return reward, done, obs

    def _searchinmap(self, actions):  # 1221

        REWARDMAP = {
            0: {0: 0, 1: 0},  # Displacement    NOTJUMP = 0    JUMP = 1
            1: {0: 0, 1: 0, 2: 0},  # Move    STAY = 0    LEFT = 1    RIGHT = 2
            2: {0: 0, 1: 0, 2: 0},  # Attack    NOTATTACK = 0    ATTACK = 1    DONWATTACK = 2
            3: {0: 0, 1: 0}  # Is_attack_then_move
        }

        reward = 0
        for i in [0, 1, 2, 3]:
            reward += REWARDMAP[i][actions[i]]

        return reward

    def _get_reward(self, knight_hp, enemy_hp, win, turnon=False):  # w1=1, w2=1

        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp
        # if hit:
        #     # print("hit")
        # print(f"hit:{hit},enemy_hp:{enemy_hp},prev_enemy_hp{self.prev_enemy_hp}")
        self.prev_knight_hp = knight_hp
        self.prev_enemy_hp = enemy_hp

        if hurt and sum(self._last_5jump) < 5 and turnon:
            # last 5 frames can't be all trying to jump
            check = self._checkiftryjump()  # return 0.5 -> 1
        else:
            check = 1
        reward = (
                - self.w1 * hurt * check
                + self.w2 * hit
            # - 0.05
            # + action_rew
        )
        if win:
            reward += 0.7
            # reward += knight_hp / 45
            print("win!!!!!")
        # print('reward:', reward)
        return reward

    def _check_done(self, knight_hp, enemy_hp):
        done = False
        win = False
        if self.prev_enemy_hp is not None:
            win = self.prev_enemy_hp < enemy_hp and not knight_hp > self.prev_knight_hp  # 敌人血量变多 and not 自己血量变多
            lose = knight_hp == 0 or knight_hp > self.prev_knight_hp  # 或者自己血量变多
            done = win or lose
            # print(f"win:{win}")

        return done, win

    def _observe(self, force_gray=False):
        """
        take a screenshot and identify enemy and knight's HP

        :param force_gray: override self.rgb to force return gray obs
        :return: observation (a resized screenshot), knight HP ?, and enemy HP:0->1
        """
        with MSS() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
        enemy_hp_bar = frame[-1, 187:826, :]
        if (np.all(enemy_hp_bar[..., 0] == enemy_hp_bar[..., 1]) and
                np.all(enemy_hp_bar[..., 1] == enemy_hp_bar[..., 2])):
            # hp bar found
            enemy_hp = (enemy_hp_bar[..., 0] < 3).sum() / len(enemy_hp_bar)
        else:
            enemy_hp = 1.
        knight_hp_bar = frame[64, :, 0]
        checkpoint1 = knight_hp_bar[self.HP_CKPT]
        checkpoint2 = knight_hp_bar[self.HP_CKPT - 1]
        knight_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
        rgb = not force_gray and self.rgb
        obs = cv2.cvtColor(frame[:672, ...],
                           (cv2.COLOR_BGRA2RGB if rgb
                            else cv2.COLOR_BGRA2GRAY))
        obs = cv2.resize(obs,
                         dsize=self.observation_space.shape[1:],
                         interpolation=cv2.INTER_AREA)
        # obs = np.expand_dims(obs, axis=0)
        # make channel first
        obs = np.rollaxis(obs, -1) if rgb else obs[np.newaxis, ...]
        return obs, knight_hp, enemy_hp

    def _find_menu(self):
        """
        locate the menu badge,
        when the badge is found, the correct game is ready to be started

        :return: the location of menu badge
        """
        monitor = self.monitor
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'] + monitor['height'] // 4,
                   monitor['width'] // 2,
                   monitor['height'] // 2)
        return pyautogui.locateOnScreen(f'locator/attuned.png',
                                        region=monitor,
                                        confidence=0.925)

    @staticmethod
    def _find_window():
        """
        find the location of Hollow Knight window

        :return: return the monitor location for screenshot
        """
        window = pyautogui.getWindowsWithTitle('Hollow Knight')
        assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'
        window = window[0]
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.moveTo(0, 0)

        geo = None
        conf = 0.9995
        while geo is None:
            geo = pyautogui.locateOnScreen('./locator/geo.png',
                                           confidence=conf)
            conf = max(0.92, conf * 0.999)
            time.sleep(0.1)
        loc = {
            'left': geo.left - 36,
            'top': geo.top - 97,
            'width': 1020,
            'height': 692
        }
        return loc

    def _checkiftryjump(self):
        """return 0.5 -> 1"""
        check = -sum(self._last_5jump[0:3]) / 6 + 1
        return check


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    # import gymnasium
    env = HKEnv()
    check_env(env, warn=True)
