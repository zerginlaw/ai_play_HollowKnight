from model import env
from model import callback
from stable_baselines3 import PPO

model = PPO.load(r"zote02models/0k_add__22539_steps.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\zote02",

                     learning_rate=3e-4,
                     n_steps=2048, batch_size=512,

                     clip_range=0.2,
                     gae_lambda=0.9,
                     n_epochs=4,
                     ent_coef=0.05, vf_coef=3
                 )
                 )

# print(model.policy)
model = model.learn(total_timesteps=int(
    8192 * 5+100
    # 10
),
    reset_num_timesteps=False,

    progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
