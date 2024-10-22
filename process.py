import math
import concurrent.futures
from dataclasses import dataclass

from ppo import train

@dataclass
class Args:
    dataset_name: str = None
    num_classes: int = 1
    decode_method: str = "loss_based"
    exp_name: str = f"ppo_ecoc_{dataset_name}_{decode_method}"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "rlECOC"
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "ECOC"
    total_timesteps: int = 50000
    learning_rate: float = 2.5e-3
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    max_columns: int = math.ceil(10 * math.log2(num_classes))
    sample_thres: int = 512

    # args to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

def run_experiment(seed, dataset_name, num_classes, decode_method):
    max_columns = math.ceil(10 * math.log2(num_classes))
    args = Args(seed=seed, dataset_name=dataset_name, num_classes=num_classes, decode_method=decode_method)
    args.max_columns = max_columns
    args.exp_name = f"ppo_ecoc_{dataset_name}_{decode_method}_seed{seed}"
    train(args)
    return f"Experiment with seed {seed} completed."

def run(dataset_name, num_classes):
    num_seeds = 10 # A total of 10 seeds are run for each dataset
    max_workers = 10  # Adjust this based on your system's capabilities

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # register experiments with each seed number
        future_to_seed = {
            executor.submit(run_experiment, seed, dataset_name, num_classes, method): seed 
            for method in ['loss_based', 'prob_loss_based'] # add decode method as needed
            for seed in range(1, num_seeds + 1)
        }
        
        # multithread experiment executer
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"Experiment with seed {seed} generated an exception: {exc}")

    print("All experiments completed.")