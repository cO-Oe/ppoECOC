import os
import random
import time
import numpy as np
import gymnasium as gym

import csv
import uuid
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from env import ECOCEnv
from agent  import Agent

def make_env(env_id, idx, capture_video, run_name, num_classes, max_columns, dataset_name, seed, decode_method):
    # environment initialize helper function
    def thunk():
        if capture_video and idx == 0:
            env = ECOCEnv(num_classes, max_columns, dataset_name, seed, decode_method)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = ECOCEnv(num_classes, max_columns, dataset_name, seed, decode_method)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
    
def train(args):
    # setup args to be initialized at runtime
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Tensorboard writer init
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # configure the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # environments init
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.num_classes, args.max_columns, args.dataset_name, args.seed, args.decode_method) for i in range(args.num_envs)],
    )

    # agent init
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # replay buffers init for PPO algorithm
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # get first state (as next_obs)
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # global score trackers init
    GLOBAL_BEST = -np.inf
    episodes_without_improvement = 0
    max_episodes_without_improvement = 1000

    # Main training loop
    for iteration in range(1, args.num_iterations + 1):
        # learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # start an episode
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            # get state
            obs[step] = next_obs
            dones[step] = next_done

            # get action and logprob
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # environment step feed back with next state and reward
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos: 
                # record the scores on episode terminations
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # record score to tensorboard
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode']['l'], global_step)

                    if info and "best_score" in info:
                        # record best score to tensorboard and update current best score
                        best_score = info['best_score']
                        writer.add_scalar("charts/episodic_best", info['best_score'], global_step)

                        if info['best_score'] >= GLOBAL_BEST:
                            GLOBAL_BEST = info['best_score']
                            save_best(args.dataset_name, args.decode_method, args.seed, info['best_metrics'], info['best_matrix'], info['best_classifiers'])
                            episodes_without_improvement = 0
                        else:
                            episodes_without_improvement += 1
                        # print(f"global_step={global_step}, episodes_without_improvement={episodes_without_improvement}")

                        if best_score == 3.0:
                            # check if best score is reached 3 (perfect score)
                            print("Early termination: Achieved perfect score of 3.0")
                            save_model(agent, optimizer, args, f"./agent/ecoc_model_{run_name}_perfect_score.pth")
                            envs.close()
                            writer.close()
                            return  # Exit the training loop

                        if episodes_without_improvement >= max_episodes_without_improvement:
                            # early training termination conditions
                            print(f"Early termination: No improvement for {max_episodes_without_improvement} episodes")
                            save_model(agent, optimizer, args, f"./agent/ecoc_model_{run_name}_early_stop.pth")
                            envs.close()
                            writer.close()
                            return 

        # calculate GAE for PPO update
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # retrieve transitions from current memory buffer
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # PPO update epochs
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # extract minibatch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # get the distribution with current agent (on-policy)
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # calculate the old and new approximated KL divergence values
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # retrieve advantages for the minibatch and normalize them if specified in the arguments
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # calculate policy loss using two methods:
                # 1. The standard policy gradient loss
                pg_loss1 = -mb_advantages * ratio
                # 2. The clipped version of the policy gradient loss to limit the policy update
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # take the maximum of the two losses, following the PPO objective to prevent large updates
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)

                # calculate value loss
                if args.clip_vloss:
                    # unclipped loss between new value predictions and returns
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2

                    # clip the predicted values to limit the update, ensuring stability
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )

                    # clipped value loss, comparing the clipped values with returns
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # use the maximum of unclipped and clipped loss, which is more conservative
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # standard MSE loss between predicted values and returns
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # calculate entropy loss, used to encourage exploration by maximizing entropy
                entropy_loss = entropy.mean()

                # combine policy loss, entropy loss, and value loss into a single objective
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # update model parameters
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # get explained variance 
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # record the statistics to tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    # save current model 
    save_model(agent, optimizer, args, f"./agent/ecoc_model_{run_name}.pth")
    envs.close()
    writer.close()
    

def save_model(agent, optimizer, args, filename):
    # save model helper
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, filename)
    print(f"Model saved to {filename}")

def load_model(filename, device):
    # loads model and environment with model name (filename)
    checkpoint = torch.load(filename, map_location=device)
    args = checkpoint['args']
    
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, False, "inference", args.num_classes, args.max_columns, args.decode_method)])
    agent = Agent(envs).to(device)  # This line uses the new Agent class
    agent.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return agent, optimizer, args, envs

def evaluate(args, name):
    # run an inference with a specific model
    print("Starting inference...")
    inference_device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    inference_agent, _, inference_args, inference_env = load_model(name, inference_device)
    inference_agent.eval()

    obs, _ = inference_env.reset()
    done = False
    total_reward = 0

    while not done:
        obs = torch.FloatTensor(obs).to(inference_device)
        with torch.no_grad():
            action, _, _, _ = inference_agent.get_action_and_value(obs)
        obs, reward, terminations, truncations, _ = inference_env.step(action.cpu().numpy())
        done = terminations.any() or truncations.any()
        total_reward += reward.item()

    print(f"Inference complete. Total reward: {total_reward}")
    inference_env.close()

def save_best(dataset_name, decode_method, seed, metrics, matrix, classifiers):
    # records the model, classifier, and metric scores with unique id

    # create a unique id for this save
    unique_id = str(uuid.uuid4())
    base_path = f"./results/{dataset_name}_{decode_method}_{seed}"
    os.makedirs(base_path, exist_ok=True)

    # open and append to the results.csv file
    csv_path = os.path.join(base_path, "results.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['id'] + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()
        
        row = {'id': unique_id, **metrics}
        writer.writerow(row)

    # create a folder with the unique_id
    save_folder = os.path.join(base_path, unique_id)
    os.makedirs(save_folder, exist_ok=True)

    # save the matrix to a txt file
    matrix_path = os.path.join(save_folder, "matrix.txt")
    np.savetxt(matrix_path, matrix, fmt='%d')

    # create a classifiers folder and save the SVM classifiers
    classifiers_folder = os.path.join(save_folder, "classifiers")
    os.makedirs(classifiers_folder, exist_ok=True)

    for i, (clf, mask) in enumerate(classifiers):
        clf_path = os.path.join(classifiers_folder, f"classifier_{i}.pkl")
        with open(clf_path, 'wb') as f:
            pickle.dump((clf, mask), f)

    print(f"Best results saved for dataset {dataset_name} with ID: {unique_id}")