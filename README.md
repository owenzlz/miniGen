# miniGen: A 2D Playground for Generative Models

Generative modeling is probability distribution mapping in essense.

<table>
  <tr>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/swiss_roll/drifting_model/drifting_model_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/gaussian_mixture/drifting_model/drifting_model_gaussian_mixture_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/moons/drifting_model/drifting_model_moons_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/circles/drifting_model/drifting_model_circles_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/checkerboard/drifting_model/drifting_model_checkerboard_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/spirals/drifting_model/drifting_model_spirals_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/pinwheel/drifting_model/drifting_model_pinwheel_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="12.5%"><img src="assets/drifting_field_distribution_matching/s_curve/drifting_model/drifting_model_s_curve_distribution_matching_process.gif" width="100%"></td>
  </tr>
</table>

This is a minimalist, modular playground for training and visualizing generative models on 2D synthetic data (source distribution → target distribution). Built for rapid experimentation, prototyping, and learning, it helps me stay current with new generative modeling ideas. The focus is on core algorithmic logic, which largely carries over to production systems; most real-world differences stem from scale and training refinements. 

## Visualization

### Distribution Matching Process

Below are visualizations showing how distribution matching evolves during training across different generative modeling paradigms/methods.

<table>
  <tr>
    <td align="center" width="20%"><sub><b>Variational<br>Autoencoder</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/variational_autoencoder/variational_autoencoder_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>GAN<br>&nbsp;</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/generative_adversarial_network/generative_adversarial_network_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Energy-Based<br>Model</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/energy_based_model/energy_based_model_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Autoregressive<br>Model</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/autoregressive/autoregressive_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Score Matching<br>(Hyv&auml;rinen)</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/score_matching_hyvarinen/score_matching_hyvarinen_swiss_roll_distribution_matching_process.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="20%"><sub><b>Score Matching<br>(NCSN)</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/score_matching_ncsn/score_matching_ncsn_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Diffusion<br>(DDPM)</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/ddpm/ddpm_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Normalizing<br>Flow</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/normalizing_flow/normalizing_flow_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Continuous<br>NF</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/continuous_normalizing_flow/continuous_normalizing_flow_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>(Rectified) Flow<br>Matching</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/flow_matching/flow_matching_swiss_roll_distribution_matching_process.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center" width="20%"><sub><b>Consistency<br>Training</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/consistency_training/consistency_training_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Shortcut<br>Model</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/shortcut_model/shortcut_model_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Improved<br>MeanFlow</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/improved_meanflow/improved_meanflow_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Alpha<br>Flow</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/alpha_flow/alpha_flow_swiss_roll_distribution_matching_process.gif" width="100%"></td>
    <td align="center" width="20%"><sub><b>Drifting<br>Model</b></sub><br><img src="assets/distribution_matching_process/swiss_roll/drifting_model/drifting_model_swiss_roll_distribution_matching_process.gif" width="100%"></td>
  </tr>
</table>


### The Score Fields (Score-based Models)

Visualizing the learned score fields over different training iterations.

<table>
  <tr>
    <td align="center" width="33%"><sub><b>Score Matching<br>(Hyv&auml;rinen)</b></sub><br><img src="assets/score_field/swiss_roll/score_matching_hyvarinen/score_matching_hyvarinen_swiss_roll_score_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>Score Matching<br>(NCSN)</b></sub><br><img src="assets/score_field/swiss_roll/score_matching_ncsn/score_matching_ncsn_swiss_roll_score_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>Diffusion<br>(DDPM)</b></sub><br><img src="assets/score_field/swiss_roll/ddpm/ddpm_swiss_roll_score_field.gif" width="100%"></td>
  </tr>
</table>


### The Velocity Fields (Flow Models)

Visualizing the learned velocity fields over different training iterations.

<table>
  <tr>
    <td align="center" width="33%"><sub><b>Continuous<br>Normalizing Flow</b></sub><br><img src="assets/velocity_field/swiss_roll/continuous_normalizing_flow/continuous_normalizing_flow_swiss_roll_velocity_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>(Rectified) Flow<br>Matching</b></sub><br><img src="assets/velocity_field/swiss_roll/flow_matching/flow_matching_swiss_roll_velocity_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>Improved<br>MeanFlow</b></sub><br><img src="assets/velocity_field/swiss_roll/improved_meanflow/improved_meanflow_swiss_roll_velocity_field.gif" width="100%"></td>
  </tr>
</table>


### The Drift Fields (Drifting Model)

Visualizing the "drifting" motion at inference time.

<table>
  <tr>
    <td align="center" width="33%"><sub><b>Swiss Roll<br>&nbsp;</b></sub><br><img src="assets/drifting_field/swiss_roll/drifting_model/drifting_model_swiss_roll_drifting_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>Gaussian Mixture<br>&nbsp;</b></sub><br><img src="assets/drifting_field/gaussian_mixture/drifting_model/drifting_model_gaussian_mixture_drifting_field.gif" width="100%"></td>
    <td align="center" width="33%"><sub><b>Checkerboard<br>&nbsp;</b></sub><br><img src="assets/drifting_field/checkerboard/drifting_model/drifting_model_checkerboard_drifting_field.gif" width="100%"></td>
  </tr>
</table>

The drifting model performs one-step generation:
- Blue dots represent randomly initialized points
- Red dots represent the drifted outputs of the model
- Green lines visualize their one-step drifting process or trajectories.

## Usage

### Training

Train any model by pointing to its config:

```bash
python train.py --config configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml
```

Override config values from the command line:

```bash
python train.py --config configs/diffusion/DDPMSampler_MLP_SwissRoll.yaml training.total_steps=5000
```

Add `--skip_save_ckpt` to skip saving checkpoints.

Configs are organized as `configs/{paradigm}/{Model}_{Arch}_{Dataset}.yaml`. Each config specifies the model type, architecture, dataset, training hyperparameters, and inference settings.

### Supported Models

Currently 18 generative modeling paradigms are supported — see `paradgims/` for the full list.

### Datasets

Nine 2D synthetic distributions: `swiss_roll`, `gaussian_mixture`, `moons`, `circles`, `checkerboard`, `spirals`, `pinwheel`, `rings`, `s_curve`.

### Outputs

Training outputs are saved to `logs/{timestamp}_{exp_name}/`:
- `train.log` — training metrics (step, loss, lr, throughput)
- `saved_images/` — sample visualizations at regular intervals
- `saved_checkpoints/` — model checkpoints
- `visualize.html` — interactive image gallery

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync
```

This installs all dependencies (PyTorch, NumPy, matplotlib, OmegaConf, tqdm) into a local `.venv`. The PyTorch wheel is platform-aware: CUDA-enabled on Linux (x86_64), CPU+MPS on macOS (Apple Silicon).

Then run training with:

```bash
uv run python train.py --config configs/flow_matching/FlowMatching_MLP_SwissRoll.yaml
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- PyTorch, NumPy, matplotlib, OmegaConf, tqdm (managed by `pyproject.toml`)
