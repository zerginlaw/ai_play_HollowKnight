from model import env
from model import callback
from stable_baselines3 import PPO

model = PPO.load(r"newenv18_actionspace_models/59k_add__36882_steps.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\gru_new18_nn",

                     learning_rate=0.0001,
                     n_steps=4096, batch_size=512,
                     clip_range=0.2,
                     gae_lambda=0.9,
                     n_epochs=3,
                     ent_coef=0.0001, vf_coef=0.5,
                 )
                 )

# print(model.policy)
model = model.learn(total_timesteps=int(
    13333 * 12
    # 10
),
    reset_num_timesteps=True,

    # progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
