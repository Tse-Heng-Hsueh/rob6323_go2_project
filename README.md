# ROB6323 Go2 Project — Isaac Lab

This repository is the starter code for the NYU Reinforcement Learning and Optimal Control project in which students train a Unitree Go2 walking policy in Isaac Lab starting from a minimal baseline and improve it via reward shaping and robustness strategies. Please read this README fully before starting and follow the exact workflow and naming rules below to ensure your runs integrate correctly with the cluster scripts and grading pipeline.

## Repository policy

- Fork this repository and do not change the repository name in your fork.  
- Your fork must be named rob6323_go2_project so cluster scripts and paths work without modification.

### Prerequisites

- **GitHub Account:** You must have a GitHub account to fork this repository and manage your code. If you do not have one, [sign up here](https://github.com/join).

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Connect to Greene

- Connect to the NYU Greene HPC via SSH; if you are off-campus or not on NYU Wi‑Fi, you must connect through the NYU VPN before SSHing to Greene.  
- The official instructions include example SSH config snippets and commands for greene.hpc.nyu.edu and dtn.hpc.nyu.edu as well as VPN and gateway options: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0#h.7t97br4zzvip.

## Clone in $HOME

After logging into Greene, `cd` into your home directory (`cd $HOME`). You must clone your fork into `$HOME` only (not scratch or archive). This ensures subsequent scripts and paths resolve correctly on the cluster. Since this is a private repository, you need to authenticate with GitHub. You have two options:

### Option A: Via VS Code (Recommended)
The easiest way to avoid managing keys manually is to configure **VS Code Remote SSH**. If set up correctly, VS Code forwards your local credentials to the cluster.
- Follow the [NYU HPC VS Code guide](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code) to set up the connection.

> **Tip:** Once connected to Greene in VS Code, you can clone directly without using the terminal:
> 1. **Sign in to GitHub:** Click the "Accounts" icon (user profile picture) in the bottom-left sidebar. If you aren't signed in, click **"Sign in with GitHub"** and follow the browser prompts to authorize VS Code.
> 2. **Clone the Repo:** Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`), type **Git: Clone**, and select it.
> 3. **Select Destination:** When prompted, select your home directory (`/home/<netid>/`) as the clone location.
>
> For more details, see the [VS Code Version Control Documentation](https://code.visualstudio.com/docs/sourcecontrol/intro-to-git#_clone-a-repository-locally).

### Option B: Manual SSH Key Setup
If you prefer using a standard terminal, you must generate a unique SSH key on the Greene cluster and add it to your GitHub account:
1. **Generate a key:** Run the `ssh-keygen` command on Greene (follow the official [GitHub documentation on generating a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key)).
2. **Add the key to GitHub:** Copy the output of your public key (e.g., `cat ~/.ssh/id_ed25519.pub`) and add it to your account settings (follow the [GitHub documentation on adding a new SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)).

### Execute the Clone
Once authenticated, run the following commands. Replace `<your-git-ssh-url>` with the SSH URL of your fork (e.g., `git@github.com:YOUR_USERNAME/rob6323_go2_project.git`).
```
cd $HOME
git clone <your-git-ssh-url> rob6323_go2_project
```
*Note: You must ensure the target directory is named exactly `rob6323_go2_project`. This ensures subsequent scripts and paths resolve correctly on the cluster.*
## Install environment

- Enter the project directory and run the installer to set up required dependencies and cluster-side tooling.  
```
cd $HOME/rob6323_go2_project
./install.sh
```
Do not skip this step, as it configures the environment expected by the training and evaluation scripts. It will launch a job in burst to set up things and clone the IsaacLab repo inside your greene storage. You must wait until the job in burst is complete before launching your first training. To check the progress of the job, you can run `ssh burst "squeue -u $USER"`, and the job should disappear from there once it's completed. It takes around **30 minutes** to complete. 
You should see something similar to the screenshot below (captured from Greene):

![Example burst squeue output](docs/img/burst_squeue_example.png)

In this output, the **ST** (state) column indicates the job status:
- `PD` = pending in the queue (waiting for resources).
- `CF` = instance is being configured.
- `R`  = job is running.

On burst, it is common for an instance to fail to configure; in that case, the provided scripts automatically relaunch the job when this happens, so you usually only need to wait until the job finishes successfully and no longer appears in `squeue`.

## What to edit

- In this project you'll only have to modify the two files below, which define the Isaac Lab task and its configuration (including PPO hyperparameters).  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py  
  - source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py
PPO hyperparameters are defined in source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/agents/rsl_rl_ppo_cfg.py, but you shouldn't need to modify them.

## How to edit

- Option A (recommended): Use VS Code Remote SSH from your laptop to edit files on Greene; follow the NYU HPC VS Code guide and connect to a compute node as instructed (VPN required off‑campus) (https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/vs-code). If you set it correctly, it makes the login process easier, among other things, e.g., cloning a private repo.
- Option B: Edit directly on Greene using a terminal editor such as nano.  
```
nano source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py
```
- Option C: Develop locally on your machine, push to your fork, then pull changes on Greene within your $HOME/rob6323_go2_project clone.

> **Tip:** Don't forget to regularly push your work to github

## Launch training

- From $HOME/rob6323_go2_project on Greene, submit a training job via the provided script.  
```
cd "$HOME/rob6323_go2_project"
./train.sh
```
- Check job status with SLURM using squeue on the burst head node as shown below.  
```
ssh burst "squeue -u $USER"
```
Be aware that jobs can be canceled and requeued by the scheduler or underlying provider policies when higher-priority work preempts your resources, which is normal behavior on shared clusters using preemptible partitions.

## Where to find results

- When a job completes, logs are written under logs in your project clone on Greene in the structure logs/[job_id]/rsl_rl/go2_flat_direct/[date_time]/.  
- Inside each run directory you will find a TensorBoard events file (events.out.tfevents...), neural network checkpoints (model_[epoch].pt), YAML files with the exact PPO and environment parameters, and a rollout video under videos/play/ that showcases the trained policy.  

## Download logs to your computer

Use `rsync` to copy results from the cluster to your local machine. It is faster and can resume interrupted transfers. Run this on your machine (NOT on Greene):

```
rsync -avzP -e 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' <netid>@dtn.hpc.nyu.edu:/home/<netid>/rob6323_go2_project/logs ./
```

*Explanation of flags:*
- `-a`: Archive mode (preserves permissions, times, and recursive).
- `-v`: Verbose output.
- `-z`: Compresses data during transfer (faster over network).
- `-P`: Shows progress bar and allows resuming partial transfers.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rob6323_go2_project` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.

## Project scope reminder

- The assignment expects you to go beyond velocity tracking by adding principled reward terms (posture stabilization, foot clearance, slip minimization, smooth actions, contact and collision penalties), robustness via domain randomization, and clear benchmarking metrics for evaluation as described in the course guidelines.  
- Keep your repository organized, document your changes in the README, and ensure your scripts are reproducible, as these factors are part of grading alongside policy quality and the short demo video deliverable.

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).

---
---

# My Modifications (ROB6323 Go2 Project)

This fork extends the provided minimal baseline Go2 walking environment in Isaac Lab by adding a custom torque controller, principled reward shaping, termination logic, and domain randomization for robustness. All changes are implemented with concise inline comments in the two allowed files:

- `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py`
- `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py`

## Summary of Major Changes (What / Why)

### Part 1 — Action Smoothness (Action-Rate Penalties)
**What I changed**
- Added an action history buffer (`last_actions`, history length = 3).
- Added action-rate and action-acceleration penalties (1st/2nd discrete derivatives).
- Added TensorBoard logging key: `rew_action_rate`.

**Why**
- Encourages smoother control signals and reduces high-frequency oscillations / jitter.

---

### Part 2 — Low-Level PD Torque Controller (Explicit Control)
**What I changed**
- Disabled Isaac Lab’s implicit actuator PD by setting actuator `stiffness=0` and `damping=0` in config.
- Implemented a manual torque-level PD controller in `_apply_action()`:
  \tau = Kp*(q_des - q) - Kd*qdot
- Clipped torques with `torque_limits` for stability and safety.
- Added torque penalty term `rew_torque` and logging.

**Why**
- Makes control behavior explicit and tunable, improves interpretability, and supports adding actuator-level effects.

---

### Part 3 — Early Termination (Base Height)
**What I changed**
- Added `base_height_min` threshold in config.
- Terminate episode if base height falls below threshold (in addition to existing contact/upsidedown termination).

**Why**
- Speeds up training by ending failed episodes early and reinforces upright locomotion.

---

### Part 4 — Raibert Heuristic Shaping + Observation Expansion
**What I changed**
- Implemented gait phase scheduler (`_step_contact_targets`) and Raibert heuristic reward (`_reward_raibert_heuristic`).
- Added 4-D clock inputs (`clock_inputs`) appended to policy observations.
- Increased `observation_space` from 48 → 52.
- Added logging key: `raibert_heuristic`.

**Why**
- Provides a principled “teacher” signal for foot placement and helps policy learn periodic gait structure.

---

### Part 5 — Posture & Motion Stabilization Penalties
**What I changed**
- Added penalties:
  - `orient`: tilt penalty via projected gravity XY
  - `lin_vel_z`: vertical bouncing penalty
  - `dof_vel`: joint velocity penalty
  - `ang_vel_xy`: roll/pitch angular velocity penalty
- Added corresponding scales in config and TensorBoard logging keys.

**Why**
- Improves gait stability and reduces unstable body motion.

---

### Part 6 — Foot Interaction Shaping (Clearance + Contact Forces)
**What I changed**
- Implemented swing-phase foot clearance shaping using gait phase.
- Implemented contact force shaping using `ContactSensorData.net_forces_w`.
- Carefully separated indices:
  - `_feet_ids` for robot kinematics (positions)
  - `_feet_ids_sensor` for contact sensor indexing (forces)
- Added logging keys: `feet_clearance`, `tracking_contacts_shaped_force`.

**Why**
- Encourages appropriate foot swing height and discourages ground contact when foot is expected to be in swing, improving stepping quality.

---

### Bonus 1 — Domain Randomization: Actuator Friction
**What I changed**
- Added per-episode randomization of actuator friction parameters in `_reset_idx`:
  - viscous friction coefficient `mu_v`
  - stiction magnitude `F_s`
- Applied friction torque in `_apply_action`:
  tau_friction = F_s * tanh(qdot / eps) + mu_v * qdot

**Why**
- Improves robustness by training policies that tolerate actuator variability (basic sim robustness strategy).

---

## Files Changed (Exactly Two)
- `rob6323_go2_env.py`: controller, observations, rewards, termination, reset randomization, logging
- `rob6323_go2_env_cfg.py`: reward scales, observation dim update, PD disabling, controller gains, termination threshold

No other project files were modified.

---
## How to Reproduce My Results (Greene HPC)

### 1. Pull latest code

    cd "$HOME/rob6323_go2_project"
    git pull

### 2. Launch training

    cd "$HOME/rob6323_go2_project"
    ./train.sh

### 3. Monitor job

    ssh burst "squeue -u $USER"

---
### Reference Runs and Rubric Coverage

To clearly demonstrate compliance with the grading rubric (Policy Quality),
I provide multiple training runs corresponding to each tutorial part and extension.
All runs were trained using the same repository structure and launch script (`./train.sh`)
on Greene HPC.

#### Tutorial Reproduction (Parts 1–4) 

The following runs incrementally implement the official tutorial components:

- **Baseline (no shaping):**  
  `logs/132488/`  
  Minimal velocity-tracking baseline provided by the starter code.

- **Part 1 – Action rate penalties:**  
  `logs/132469/`  
  Adds action smoothness regularization using first- and second-order action differences.

- **Part 2 – Low-level PD torque controller:**  
  `logs/132639/`  
  Disables implicit actuator PD and applies explicit torque-level PD control.

- **Part 3 – Early termination (base height):**  
  `logs/132657/`  
  Terminates episodes when base height drops below a threshold.

- **Part 4 – Raibert heuristic + clock inputs:**  
  `logs/132672/`  
  Adds gait phase scheduling, Raibert foot placement shaping, and 4-D clock inputs.

Together, these runs reproduce the tutorial Parts 1–4 as required.

---

#### Walking / Trotting Gait, Stability, Command Following 

- **Part 5 – Posture and motion stabilization:**  
  `logs/133245/`  
  Adds penalties for body tilt, vertical bouncing, joint velocity, and roll/pitch rates,
  resulting in a stable walking gait with low oscillations and reasonable base height.

- **Part 6 – Foot interaction shaping:**  
  `logs/133448/`  
  Introduces swing-phase foot clearance shaping and contact-force penalties using
  ContactSensor data. Produces a clear, periodic footfall pattern (walking / trotting),
  avoids hopping or pacing, and improves ground contact behavior.

These runs demonstrate:
- Stable base attitude and height
- Clear periodic gait
- Reliable command following for $(v_x, v_y, \dot{\psi})$
- Proper alignment of command (green) and actual velocity (blue) arrows in videos

---

#### Action Regularization and Robustness (Bonus) 

- **Bonus 1 – Actuator friction domain randomization:**  
  `logs/133470/`  
  Adds per-episode randomization of viscous and stiction friction coefficients at the
  actuator level. Improves robustness while maintaining smooth torques and motions.

Torque magnitude penalties use a small scale (−0.0001), resulting in visually smooth
and well-regularized actions.

---

#### Recommended Evaluation Run

For grading and quick verification, the following run best reflects overall policy quality:

- **Recommended run log dir:**  
  `logs/133448/`  and `logs/133470/`
- **Representative checkpoint:**  
  `model_<EPOCH>.pt` (from the end of training)  
- **Notes:**  
  Stable walking gait, low base oscillation, smooth actions, and accurate command tracking.

All referenced runs can be reproduced by pulling this repository and executing `./train.sh`
on Greene HPC as described above.

