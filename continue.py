from model import env
from model import callback
from stable_baselines3 import PPO
import math


def learning_rate_cosine(progress_remaining, max_lr=3e-4, min_lr=7.5e-5):  # 这是一个cos函数，先上升后下降，只有一个极大值
    return math.cos(-2 * math.acos(min_lr / max_lr) * progress_remaining + math.acos(min_lr / max_lr)) * max_lr


model = PPO.load(r"zote07models/0k_add__59421_steps.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\zote08",

                     learning_rate=learning_rate_cosine,
                     n_steps=4096, batch_size=256,
                     clip_range=0.2,

                     gae_lambda=0.95,
                     n_epochs=4,
                     ent_coef=0.01, vf_coef=0.5
                 )
                 )

# print(model.policy)
model = model.learn(total_timesteps=int(
    2048 * 10 + 100
    # 10
),
    reset_num_timesteps=False,

    progress_bar=False,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
"""
zote05 存档  最好能打到boss的血量0.56差不多
28k_add__32784_steps_baseline.zip 有打到0.4过

"""
