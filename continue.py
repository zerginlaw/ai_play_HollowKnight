from model import env
from model import callback
from stable_baselines3 import PPO

model = PPO.load(r"zote01models/0k_add__81933_steps.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\zote01",

                     learning_rate=0.0006,
                     n_steps=8192, batch_size=512,

                     clip_range=0.2,
                     gae_lambda=0.9,
                     n_epochs=10,
                     ent_coef=0.05, vf_coef=0.7
                 )
                 )

# print(model.policy)
model = model.learn(total_timesteps=int(
    8192 * 2 * 2
    # 10
),
    reset_num_timesteps=False,

    progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
