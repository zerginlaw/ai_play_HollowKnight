from model import env
from model import callback
from stable_baselines3 import PPO

model = PPO.load(r"zote03models/0k_add__81960_steps_baseline.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\zote04",

                     learning_rate=1e-3,
                     n_steps=1024, batch_size=512,

                     clip_range=0.2,
                     gae_lambda=0.9,
                     n_epochs=2,
                     ent_coef=0.5, vf_coef=0.1
                 )
                 )

# print(model.policy)
model = model.learn(total_timesteps=int(
    8192 * 5+100
    # 10
),
    reset_num_timesteps=True,

    progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
