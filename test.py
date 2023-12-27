from model import env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

police_to_evaluate = {
    1: "zote01models/0k_add__98319_steps.zip",# 1 __mean_reward 2.7 std_reward 3.4719511005261072
    # 2: "newenv18_actionspace_models/201k_add__204825_steps.zip",# 2 __mean_reward 2.688888888888889 std_reward 3.1221233673555653
    # 3: "newenv18_actionspace_models/201k_add__163860_steps.zip",
    # 4: "newenv18_actionspace_models/0k_add__63550_steps.zip",
}
# 加载训练好的模型
for x, name in police_to_evaluate.items():
    model = PPO.load(name, env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=90, return_episode_rewards=False,
                                              deterministic=True)
    print(x, "__mean_reward", mean_reward, "std_reward", std_reward)

"""zote01models/0k_add__106512_steps.zip
boss_health 0.7480438184663537
boss_health 0.8403755868544601
boss_health 0.7934272300469484
boss_health 0.7480438184663537
boss_health 0.8403755868544601
boss_health 0.8857589984350548
boss_health 0.8857589984350548
boss_health 0.7480438184663537"""