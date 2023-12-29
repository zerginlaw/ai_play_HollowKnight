from model import env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

police_to_evaluate = {1: "zote01models/0k_add__227385_steps_baseline.zip",
                      # 2: "newenv18_actionspace_models/95k_add__16388_steps.zip",
                      # 3: "newenv18_actionspace_models/95k_add__32776_steps.zip",
                      # 4: "newenv18_actionspace_models/0k_add__63550_steps.zip",
                      }
# 加载训练好的模型
for x, name in police_to_evaluate.items():
    model = PPO.load(name, env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=90, return_episode_rewards=False,
                                              deterministic=True)
    print(x, "__mean_reward", mean_reward, "std_reward", std_reward)
"""
"newenv18_actionspace_models/0k_add__59450_steps_baseline.zip" 2.13


"""
