# MiniMax-M2.5 SWE

This example guide you to 


## Requirements

you need to have access to a slurm cluster with at least 4 nodes to run this example.  Each nodes must have a shared filesystem to communicate. In this guide we assume that the nfs is mounted on `/shared`, you can change it to your own path.

You also need to have prime-rl clone on your cluster into the shared filesystem.

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git /shared/prime-rl
cd /shared/prime-rl
uv sync --all-extras
```

You might also want to create a .env inside the prime-rl directory to store the environment variables that might be use during training like wandb and huggingface tokens. The .env file will be automatically source during training.

```bash
touch .env
```

```bash
echo "WANDB_API_KEY=your_wandb_api_key" >> .env
echo "HUGGINGFACE_TOKEN=your_huggingface_token" >> .env
```

### sandbox

The [mini-swe-agent-plus](https://github.com/PrimeIntellect-ai/sandbox-mini-swe-agent-plus) environments is configured to use Prime Intellect Sandboxes. You can find more information about the sandboxes [here](https://docs.primeintellect.ai/sandboxes/overview).

You will need to create a sandbox account and get the credentials from the sandbox team into the .env file.

Alternativly you can adapt the code of the environment to use your own sandbox implementation.


## Tmux session

we recommand using the tmux helper to start the run and look at the logs.

from your slurm head node:

```bash
bash scripts/slurm_tmux.sh minimax-swe /shared/outputs/minimax-swe
```

you can then attach to it by doing `tmux attach -t minimax-swe`.

## Start the run


run the following command to start the RL training:

PS: if using the tmux helper, you can run the command in the `Terminal` (window 0) pane and look at the logs in the `Logs` (window 1) pane.

```bash
uv run rl @ examples/minimax-swe/rl.toml --output-dir /shared/outputs/minimax-swe
```

output of the command
```
XXX:XX:XX    INFO Wrote subconfigs to /shared/outputs/minimax-swe/configs [rl.py::515]
XXX:XX:XX    INFO Wrote SLURM script to /shared/outputs/minimax-swe/rl.sbatch [rl.py::534]
XXX:XX:XX    INFO Submitting: sbatch /shared/outputs/minimax-swe/rl.sbatch [rl.py::540]
XXX:XX:XX SUCCESS Submitted batch job YYYY

Logs:
  Trainer:          tail -F /shared/outputs/minimax-swe/slurm/latest_train_node_rank_0.log
  Orchestrator:     tail -F /shared/outputs/minimax-swe/slurm/latest_orchestrator.log
  Inference:        tail -F /shared/outputs/minimax-swe/slurm/latest_infer_node_rank_0.log
  Envs:             tail -F /shared/outputs/minimax-swe/logs/envs/*/*/*.log
   Train:           tail -F /shared/outputs/minimax-swe/logs/envs/train/*/*.log
    swe:           tail -F /shared/outputs/minimax-swe/logs/envs/train/swe/*.log 
```


