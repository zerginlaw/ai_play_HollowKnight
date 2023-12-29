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

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.
rollouting = False
lock = threading.Lock()

# MIN_DASH_TIME = 0.6
# BASCK_DASH_TIME = 1.5 # 这是在wiki查到的，增加0.15s
MIN_DASH_TIME = 0.75
BASCK_DASH_TIME = 1.65
KEYMAP = ["a", "s", "d", "j", "k", "l"]
INITIAL_ACTION = [0, 0, 0]


class Actions(enum.Enum):
    pass


class Displacement(Actions):
    NOTJUMP = 0  # try to not jump
    JUMP = 1  # try to jump in next 0.37s
    # DASH = 2


class Move(Actions):
    # STAY = "STAY"
    LEFT = 2
    RIGHT = 3


class Attack(Actions):
    # NOTATTACK = 0
    ATTACK = 4
    DONWATTACK = 5
    # UPATTACK = "UPATTACK"
    # SPELL = 2


class DashOrNotAction(Actions):
    NOTDASH = 6
    DASH = 7


class AttackOrNot(Actions):
    ATTACK = 8  # 正常情况隔1帧攻击
    NOTATTACK = 9  # 尝试这一帧不攻击


# class Move_then_attack(Actions):
#     """加入可以不攻击之后，这个也不是必须的了"""
#     move_then_attack = "move_then_attack"
#     attack_then_move = "attack_then_move"


class DashState(enum.Enum):
    NODASH = 0
    WHITEDASH = 1
    BLACKDASH = 2


def upattack(able_attack):
    if able_attack:
        pyautogui.keyDown("w")
        pyautogui.press("j")
        pyautogui.keyUp("w")


def downattack(able_attack):
    if able_attack:
        pyautogui.keyDown("s")
        pyautogui.press("j")
        pyautogui.keyUp("s")


def attack(able_attack):
    if able_attack:
        pyautogui.press("j")


class HKEnv(gymnasium.Env):
    metadata = {"render_modes": []}
    render_mode = ""
    reward_range = (-float("inf"), float("inf"))

    HP_CKPT = np.array([52, 91, 129, 169, 207, 246, 286, 324, 363], dtype=int)

    def __init__(self, rgb=True, w1=1, w2=1.7, time_punishment=0.05, boss="zote"):
        self.boss = boss
        self.info = dict(boss_health=None, action_list=[0] * 20)
        self.w1 = w1  # 被打
        self.w2 = w2  # 击中

        self.time_punishment = time_punishment
        self.rgb = rgb

        # original=len(Displacement) * (len(Attack) + 4 * len(Attack)) # 20

        self.action_space = gymnasium.spaces.MultiDiscrete(
            [len(Displacement) * len(Move) * len(Attack), len(DashOrNotAction), len(AttackOrNot)]
        )  # [0,7],[0,1],[0,1]
        if self.rgb:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 81, 81), dtype=np.uint8)
        else:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(1, 81, 81), dtype=np.uint8)

        self.monitor = self._find_window()
        # self._last_actions = INITIAL_ACTION
        self._prev_time = None
        self._last_move = Move.LEFT
        self.prev_enemy_hp = None
        self.prev_knight_hp = None
        self.gametime = 1  # 用于转换boss时计数，仅用于骨钉切换，如小姐姐
        self.able_a = 1
        self.dash_state = DashState.BLACKDASH
        self._last_dash_time = None
        self._last_black_dash_time = None

    def lives(self):
        """返回小骑士当前血量，在step之后被调用，即这0.15s动作之后的血量"""
        return self.prev_knight_hp

    def step(self, actions):  # [len(Displacement), len(Move), len(Attack)]
        """info 正常时候是空，9滴血结束时为boss剩余血量，打赢时是“0”
        这个模块的逻辑：
        检查这是否是暂停后的第一次调用

        记录现在的时间
        采取动作
        等待
        观察血条信息

        动作空间说[0,7],[0,1],[0,1]

        """
        # print("env begin step")
        self._check_if_esc()

        now = time.time()
        action_list = self._take_action(actions, now)  # 执行动作,并返回
        # list((if_not_attack, dash, if_jump, if_move_right, if_downattack))
        self._wait_to_see(now)  # 等待直到0.15s 经过测试，每一次实际差不多在0.153到0.157s之间波动

        reward, done, obs, win, enemy_hp, lose = self._get_this_result(actions)
        info = self._make_info(win=win, lose=lose, enemy_hp=enemy_hp, action_list=action_list)
        truncated = win
        if done:
            self.close()
        return obs, reward, done, truncated, info

    def _wait_to_see(self, now, steptime=0.15):
        t = steptime - (now - self._prev_time)  # Attack interval
        # print(f"t should be +，remaining:{t}")
        if t > 0:
            # print("还差",t)
            time.sleep(t)
        else:
            print(f"delayed:{-t}")
        self._prev_time = time.time()  # 记录时间，主要是为了_wait_to_see的间隔，同时也为计算白冲黑冲等使用

    def _check_if_esc(self):
        global rollouting
        try:
            if rollouting:
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
                time.sleep(0.5)
        except AssertionError:
            warnings.warn("do not find esc")
            self.close()
            return self._observe()[0], 0, True, True, {}

    def _make_info(self, win, lose, enemy_hp, action_list):

        self.info["action_list"] += action_list
        self.info["action_list"] = self.info["action_list"][-20:]

        if win:
            self.info["boss_health"] = 0
        elif lose:
            self.info["boss_health"] = enemy_hp
            print("boss_health", enemy_hp)
        else:
            self.info["boss_health"] = None

        return self.info

    def reset(self, seed=None, options=None, changeboss=False):
        super().reset(seed=seed)
        # print("env begin reset")
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
        if self.boss == "zote":
            time.sleep(6)

        self.prev_knight_hp, self.prev_enemy_hp = len(self.HP_CKPT), 1.
        self._prev_time = time.time()
        self._last_move = Move.LEFT
        self.able_a = 1
        self.dash_state = DashState.BLACKDASH
        observe = self._observe()[0]
        self._last_dash_time = None
        self._last_black_dash_time = None
        self.info["action_list"] = [0] * 20
        return observe, {}

    @staticmethod
    def allkeyup():
        for key in KEYMAP:
            pyautogui.keyUp(key)

    def close(self):
        """
        do any necessary cleanup on the interaction
        should only be called before or after an episode
        """
        self.allkeyup()

        gc.collect()

    @staticmethod
    def _get_action(actions):
        if_not_attack = actions[2]
        dash = actions[1]  # 冲刺放在最后处理,0,1,2
        actions = actions[0]  # 一共12个动作，取证范围为[0,11]
        if_jump, act = divmod(actions, 4)
        if_move_right, if_downattack = divmod(act, 2)
        return if_not_attack, dash, if_jump, if_move_right, if_downattack

    def _take_action(self, actions, now):
        """见Xmind思维导图
        if_not_attack=1不攻击
        dash=1 冲刺
        """
        # 先处理跳跃

        (if_not_attack,
         dash,
         if_jump,
         if_move_right,
         if_downattack) = self._get_action(actions)  # 解包

        if if_not_attack:
            self.able_a = 0  # 这里会影响攻击和下劈 # 这一帧不攻击的话，下一帧一定能攻击
        if dash:  # 如果这一帧是冲刺，则跳跃没有意义
            self._do_move(if_move_right)
            self._do_attack(if_downattack)
            self._do_dash(dash, now)
        else:  # 不冲刺
            self._do_jump(if_jump)  # 先跳跃，因为出刀一小段时间不能二段跳
            self._do_move(if_move_right)
            self._do_attack(if_downattack)

        self.able_a = 1 - self.able_a  # 能攻击则重置
        return list((if_not_attack, dash, if_jump, if_move_right, if_downattack))

    @staticmethod
    def _do_move(if_move_right):
        if if_move_right:
            pyautogui.keyUp("a")
            pyautogui.keyDown("d")
        else:  # 往左走
            pyautogui.keyUp("d")
            pyautogui.keyDown("a")

    def _do_attack(self, if_downattack):
        if if_downattack:
            downattack(self.able_a)
        else:
            attack(self.able_a)

    @staticmethod
    def _do_jump(ifjump):
        if ifjump:
            pyautogui.keyDown("k")
        else:
            pyautogui.keyUp("k")

    def _do_dash(self, dash, now):
        """dash_state是指现在能不能冲刺"""
        if dash:
            if self.dash_state == DashState.WHITEDASH:
                pyautogui.press("l")
                self._last_dash_time = now
            if self.dash_state == DashState.BLACKDASH:
                pyautogui.press("l")
                self._last_dash_time = now
                self._last_black_dash_time = now

    def _get_this_result(self, actions):

        # self._last_actions = actions

        obs, knight_hp, enemy_hp = self._observe()  # knight HP都是整数,  enemy HP:0->1；注意这个是比较占用时间的

        done, win, lose = self._check_done(knight_hp, enemy_hp)

        reward = 0
        reward += self._check_dash_state(actions)  # 修改下一回合的dash_state,并返回dash相关reward
        reward += self._get_reward(knight_hp, enemy_hp, win, actions)  # 返回血量相关reward

        # reward -= self.time_punishment  # 时间惩罚
        return reward, done, obs, win, enemy_hp, lose

    def _get_reward(self, knight_hp, enemy_hp, win, actions):  # w1=1, w2=1

        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp
        coef = 1
        if hurt:
            coef = self.prev_knight_hp - knight_hp
        self.prev_knight_hp = knight_hp
        self.prev_enemy_hp = enemy_hp
        reward = (
                - self.w1 * hurt * coef
                + self.w2 * hit
        )
        if actions[2]:  # 1是这一帧不攻击，0是攻击
            reward -= 0.25
        else:
            reward += 0.25
        if win:
            print("win!!!!!")
        # print('reward:', reward)
        return reward

    def _check_done(self, knight_hp, enemy_hp):
        done = False
        win = False
        lose = False
        if self.prev_enemy_hp is not None:
            win = self.prev_enemy_hp < enemy_hp and not knight_hp > self.prev_knight_hp  # 敌人血量变多 and not 自己血量变多
            lose = knight_hp == 0 or knight_hp > self.prev_knight_hp  # 或者自己血量变多
            done = win or lose
            # if lose:
            #     print(f"lose")
            # print(f"win:{win}")

        return done, win, lose

    def _observe(self, force_gray=False):
        """
        take a screenshot and identify enemy and knight's HP

        :param force_gray: override self.rgb to force return gray obs
        :return: observation (a resized screenshot), knight HP ?, and enemy HP:0->1
        """
        with MSS() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
        enemy_hp_bar = frame[-1, 187:826, :]  #
        if (np.all(enemy_hp_bar[..., 0] == enemy_hp_bar[..., 1]) and
                np.all(enemy_hp_bar[..., 1] == enemy_hp_bar[..., 2])):
            # hp bar found
            enemy_hp = (enemy_hp_bar[..., 0] < 4).sum() / len(enemy_hp_bar)
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

    def _check_dash_state(self, actions):
        now_time = self._prev_time
        if actions[1] and self.dash_state == DashState.BLACKDASH:  # 黑冲
            self.dash_state = DashState.NODASH  # 是下一回合的状态
            self._last_dash_time = now_time
            return 2
        elif actions[1] and self.dash_state == DashState.WHITEDASH:  # 白冲
            self.dash_state = DashState.NODASH  # 是下一回合的状态
            self._last_dash_time = now_time
            return -1

        elif self.dash_state == DashState.NODASH:
            if now_time - self._last_dash_time > MIN_DASH_TIME:
                self.dash_state = DashState.WHITEDASH

        elif self.dash_state == DashState.WHITEDASH:
            if now_time - self._last_black_dash_time > BASCK_DASH_TIME:
                self.dash_state = DashState.BLACKDASH

        return 0


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    # import gymnasium
    env = HKEnv()
    check_env(env, warn=True)
