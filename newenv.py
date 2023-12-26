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

INITIAL_ACTION = [0, 0, 0]


class Actions(enum.Enum):
    pass


class Displacement(Actions):
    NOTJUMP = "NOTJUMP"  # try to not jump
    JUMP = "JUMP"  # try to jump in next 0.37s
    # DASH = 2


class Move(Actions):
    STAY = "STAY"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Attack_or_not(Actions):
    ATTACK = "ATTACK"  # 正常情况隔1帧攻击
    NOTATTACK = "NOTATTACK"  # 尝试这一帧不攻击


class Attack(Actions):
    # NOTATTACK = 0
    ATTACK = "ATTACK"
    DONWATTACK = "DONWATTACK"
    # UPATTACK = "UPATTACK"
    # SPELL = 2


class Dash_or_not(Actions):
    DASH = "DASH"
    NOTDASH = "NOTDASH"


class Dash_state(enum.Enum):
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


def takemove(last_move, this_move):
    if last_move == this_move:
        pass
    if last_move == Move.STAY:
        if this_move == Move.LEFT:
            pyautogui.keyDown("a")
        elif this_move == Move.RIGHT:
            pyautogui.keyDown("d")
    elif last_move == Move.LEFT:
        pyautogui.keyUp("a")
        if this_move == Move.RIGHT:
            pyautogui.keyDown("d")
    elif last_move == Move.RIGHT:
        pyautogui.keyUp("d")
        if this_move == Move.LEFT:
            pyautogui.keyDown("a")


class HKEnv(gymnasium.Env):
    metadata = {"render_modes": []}
    render_mode = ""
    reward_range = (-float("inf"), float("inf"))

    DISPLACEMENT_KEYMAP = {1: "k"}
    MOVE_KEYMAP = {1: "a", 2: "d"}
    ATTACK_KEYMAP = {1: "j"}
    DONWATTACK_KEYMAP = {2: "s"}

    HP_CKPT = np.array([52, 91, 129, 169, 207, 246, 286, 324, 363], dtype=int)

    def __init__(self, rgb=True, w1=1, w2=1.5, time_punishment=0.05, boss="zote"):
        self.boss = boss
        self.info = {}
        self.w1 = w1  # 被打
        self.w2 = w2  # 击中

        self.time_punishment = time_punishment
        self.rgb = rgb

        # original=len(Displacement) * (len(Attack) + 4 * len(Attack)) # 20

        self.action_space = gymnasium.spaces.MultiDiscrete(
            [len(Displacement) * (len(Attack) + 4 * len(Attack)), len(Dash_or_not), len(Attack_or_not)]
        )  # 第二个维度是（不动*攻击上劈下劈）+（左移右移*攻击上劈下劈*先后）
        if self.rgb:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 81, 81), dtype=np.uint8)
        else:
            self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(1, 81, 81), dtype=np.uint8)

        self.monitor = self._find_window()
        self._last_actions = INITIAL_ACTION
        self._prev_time = None
        self._last_ifjump = 0
        self._last_move = Move.STAY
        self.prev_enemy_hp = None
        self.prev_knight_hp = None
        self.gametime = 1  # 用于转换boss时计数
        self.able_a = 1
        self.dash_state = Dash_state.BLACKDASH
        self._last_dash_time = None
        self._last_black_dash_time = None

    def lives(self):
        """返回小骑士当前血量，在step之后被调用，即这0.15s动作之后的血量"""
        return self.prev_knight_hp

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

    def step(self, actions, steptime=0.15):  # [len(Displacement), len(Move), len(Attack)]
        """info 正常时候是空，9滴血结束时为boss剩余血量，打赢时是“0”"""
        try:
            if rollouting:
                self._checkifthinking()
                time.sleep(0.5)
        except AssertionError:
            warnings.warn("do not find esc")
            self.close()
            return self._observe()[0], 0, True, True, {}
        now = time.time()
        self._take_action(actions, now)  # it takes time to ensure 0.3s existed before last time taking actions

        t = steptime - (now - self._prev_time)  # Attack interval
        # print(f"t should be +，remaining:{t}")
        if t > 0:
            # print("还差",t)
            time.sleep(t)

            # busy_sleep(t)

        else:
            print(f"delayed:{-t}")

        self._prev_time = time.time()  # 经过测试，每一次实际差不多在0.153到0.157s之间波动

        # busy_sleep(0.006)  # it takes time to see how much the hp changes,then returen rew,done,obs
        reward, done, obs, win, enemy_hp, lose = self._get_this_result(actions)

        info = {"boss_health": None}
        if win:
            info = {"boss_health": 0}
        elif lose:
            info = {"boss_health": enemy_hp}
        self.info = info
        truncated = win
        if done:
            self.close()

        return obs, reward, done, truncated, info

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

        if self.boss == "zote":
            time.sleep(6)

        self.prev_knight_hp, self.prev_enemy_hp = len(self.HP_CKPT), 1.
        self._prev_time = time.time()
        observe = self._observe()[0]
        return observe, {}

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

        self._last_actions = INITIAL_ACTION
        self._prev_time = None
        self._last_ifjump = 0
        self._last_move = Move.STAY
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.able_a = 1
        self.dash_state = Dash_state.BLACKDASH
        self._last_dash_time = None
        self._last_black_dash_time = None

        gc.collect()

    def _take_action(self, actions, now):
        """self.action_space = gymnasium.spaces.Discrete(
            len(Displacement) * (len(Attack) + 4 * len(Attack))) # 第二个维度是（不动*攻击上劈下劈）+（左移右移*攻击上劈下劈*先后）
            以下为 len(Displacement)=2，只有跳和不跳的情形
            actions是一个数字0,1,2-->0,2,4这个偶数是不跳，1，3奇数是跳
            actions[2]里面1是不攻击"""
        # 先处理跳跃
        if_not_attack = actions[2]
        dash = actions[1]  # 冲刺放在最后处理,0,1,2
        actions = actions[0]
        act, ifjump = divmod(actions, 2)

        if if_not_attack:
            self.able_a = 0  # 这一帧不攻击的话，下一帧一定能攻击
        #     print("不攻击")
        # else:
        #     print("攻击")

        if ifjump == self._last_ifjump:
            pass
        elif list(Displacement)[ifjump] == Displacement.NOTJUMP:  # 跳变不跳
            pyautogui.keyUp("k")
        else:  # 不跳变跳
            pyautogui.keyDown("k")
        self._last_ifjump = ifjump

        # 制作一个表
        x, y = divmod(act, 4)  # y是余数[0,3]
        move_then_attack, if_move_right = divmod(y, 2)  # y是余数[0,3] 。if_move_right只有不是“不动”才用到

        if x == len(Attack):  # 不动
            takemove(self._last_move, Move.STAY)
            self._last_move = Move.STAY
            if list(Attack)[y] == Attack.ATTACK:
                attack(self.able_a)
            elif list(Attack)[y] == Attack.DONWATTACK:
                downattack(self.able_a)
            elif list(Attack)[y] == Attack.UPATTACK:
                upattack(self.able_a)
            else:  # 不攻击
                pass
        elif x == len(Attack) - 1:  # 攻击与左右移动与顺序

            if move_then_attack:
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT
                attack(self.able_a)

            else:
                attack(self.able_a)
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT

        elif x == len(Attack) - 2:  # 下劈与左右移动与顺序
            if move_then_attack:
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT
                downattack(self.able_a)

            else:
                downattack(self.able_a)
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT

        elif x == len(Attack) - 3:  # 上劈与左右移动与顺序
            if move_then_attack:
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT
                upattack(self.able_a)

            else:
                upattack(self.able_a)
                takemove(self._last_move, Move.RIGHT if if_move_right else Move.LEFT)
                self._last_move = Move.RIGHT if if_move_right else Move.LEFT

        self.able_a = 1 - self.able_a

        self.dash(dash, now)

    def dash(self, dash, now):
        """dash_state是指现在能不能冲刺"""
        if dash:
            if self.dash_state == Dash_state.WHITEDASH:
                self._last_dash_time = now
                pyautogui.press("l")
            if self.dash_state == Dash_state.BLACKDASH:
                self._last_dash_time = now
                self._last_black_dash_time = now
                pyautogui.press("l")

    def _get_this_result(self, actions):

        self._check_dash_state(actions)  # 修改下一回合的dash_state
        self._last_actions = actions

        obs, knight_hp, enemy_hp = self._observe()  # knight HP ?,  enemy HP:0->1

        done, win, lose = self._check_done(knight_hp, enemy_hp)
        reward = self._get_reward(knight_hp, enemy_hp, win, actions)

        # reward -= self.time_punishment  # 时间惩罚
        return reward, done, obs, win, enemy_hp, lose

    def _get_reward(self, knight_hp, enemy_hp, win, actions):  # w1=1, w2=1

        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp
        coef = 1
        if hurt:
            coef = self.prev_knight_hp - knight_hp

        # if hit and self.able_a == 1:  # 击中，说明这回合是出刀，然后able_a经过转换现在应该是0，所以如果现在是1的话是有问题的
        #     print("reset able_attack")# 经过观察，似乎是显示血条mod的延迟，那就是没法改变的，反而不应该修正
        #     self.able_a = 0

        # print(f"hit:{hit},enemy_hp:{enemy_hp},prev_enemy_hp{self.prev_enemy_hp}")
        self.prev_knight_hp = knight_hp
        self.prev_enemy_hp = enemy_hp

        reward = (
                - self.w1 * hurt * coef
                + self.w2 * hit
            # - 0.05
            # + action_rew
        )
        if actions[2]:  # 1是这一帧不攻击，0是攻击
            reward -= 0.3
        else:
            reward += 0.3
        if win:
            # reward += knight_hp / 45
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

    def _check_dash_state(self, actions):
        now_time = self._prev_time
        if actions[1] == Dash_or_not.DASH and (
                self.dash_state == Dash_state.BLACKDASH or self.dash_state == Dash_state.WHITEDASH):

            self.dash_state = Dash_state.NODASH  # 是下一回合的状态
            self._last_dash_time = now_time

        elif self.dash_state == Dash_state.NODASH:
            if now_time - self._last_dash_time > MIN_DASH_TIME:
                self.dash_state = Dash_state.WHITEDASH

        elif self.dash_state == Dash_state.WHITEDASH:
            if now_time - self._last_black_dash_time > BASCK_DASH_TIME:
                self.dash_state = Dash_state.BLACKDASH


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    # import gymnasium
    env = HKEnv()
    check_env(env, warn=True)
