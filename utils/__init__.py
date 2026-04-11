"""
Utilities for training and sampling.

Modules:
- builders_utils: Create models, processes, schedules, and samplers
- cfg_utils: Classifier-free guidance wrappers
- checkpoint_utils: Save and load checkpoints
- log_utils: Training logger
- sampling_utils: Core sampling function
- visualization_utils: Sample visualization during training
"""

from .builders_utils import (
    create_diffusion_schedule,
    create_generative_process,
    create_model,
    create_optimizer,
    create_sampler,
    SamplerType,
)
from .cfg_utils import CFGWrapper, CFGWrapperBatched
from .checkpoint_utils import save_checkpoint, load_checkpoint
from .log_utils import TrainingLogger
from .sampling_utils import sample
from .visualization_utils import (
    compute_axis_limits,
    get_target_samples,
    visualize_synthetic2d,
    visualize_distribution,
    update_visualization_html,
    images_to_video,
)

__all__ = [
    # Builder functions
    "create_diffusion_schedule",
    "create_generative_process",
    "create_model",
    "create_optimizer",
    "create_sampler",
    "SamplerType",
    # CFG
    "CFGWrapper",
    "CFGWrapperBatched",
    # Checkpoint functions
    "save_checkpoint",
    "load_checkpoint",
    # Logging
    "TrainingLogger",
    # Sampling
    "sample",
    # Visualization
    "get_target_samples",
    "visualize_synthetic2d",
    "visualize_distribution",
    "update_visualization_html",
    "images_to_video",
]
