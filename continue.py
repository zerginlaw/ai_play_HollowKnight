from gru import env
from gru import callback
from stable_baselines3 import PPO

model = PPO.load(r"stack4_new_models/190k_add__19000_steps.zip",
                 env=env,
                 custom_objects=dict(
                     tensorboard_log=r"logs\gru_new1_nn",

                     learning_rate=0.0006,
                     n_steps=1800,
                     batch_size=900,
                     clip_range=0.2,
                     n_epochs=2,
                     ent_coef=0.2,
                     vf_coef=0.5,
                     normalize_advantage=True
                 )
                 )

model = model.learn(total_timesteps=int(
    13333 * 12
    # 10
),
    reset_num_timesteps=True,

    # progress_bar=True,
    callback=callback
)

# model.save(r"D:\HollowKnight_cpu\models\new_nn_7.zip")
print(model.policy)

