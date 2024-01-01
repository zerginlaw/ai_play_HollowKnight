from model import env
from model import callback
from stable_baselines3 import PPO
model = PPO.load(r"zote05models/0k_add__100401_steps_baseline.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\zote05",
                     max_grad_norm=0.01,
                     learning_rate=8e-5,
                     n_steps=2048, batch_size=256,
                     clip_range=0.2,
                     gamma=0.8,
                     gae_lambda=0.9,
                     n_epochs=6,
                     ent_coef=0.001, vf_coef=3
                 )
                 )


# print(model.policy)
model = model.learn(total_timesteps=int(
    8192 * 10 + 100
    # 10
),
    reset_num_timesteps=False,

    progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
"""
zote05 存档  最好能打到boss的血量0.56差不多


"""