"""Load attacker code and instantiate appropriate objects."""
import torch

from .optimization_based_attack import OptimizationBasedAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .optimization_with_label_attack import OptimizationJointAttacker
from .optimization_permutation_attack import OptimizationPermutationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker, DecepticonAttacker, AprilAttacker
from .recursive_attack import RecursiveAttacker
from .optimized_and_recursive_attack import Optimization_and_recursive_attacker
from .optimized_with_rgap_feature_attack import Optimization_with_grap_feature_attacker
from .optimization_GAN_attack import AdamReconstructor
from .optimization_GAN_attack import NGReconstructor
from .optimization_SR_attack import OptimizationSRAttacker
from .optimization_diffusion import OptimizationDiffusionAttacker
from .optimization_based_attack_group import OptimizationGroupAttacker
from .optimization_based_attack_batch import OptimizationBasedAttackerBatch

def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "multiscale":
        attacker = MultiScaleOptimizationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "analytic":
        attacker = AnalyticAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "april-analytic":
        attacker = AprilAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "imprint-readout":
        attacker = ImprintAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "decepticon-readout":
        attacker = DecepticonAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "recursive":
        attacker = RecursiveAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "joint-optimization":
        attacker = OptimizationJointAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "permutation-optimization":
        attacker = OptimizationPermutationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "optimization-recursive":
        attacker = Optimization_and_recursive_attacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == 'optimization_with_grap_feature':
        attacker = Optimization_with_grap_feature_attacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == 'optimization_GAN_Adam':
        attacker = AdamReconstructor(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == 'optimization_GAN_CMA':
        attacker = NGReconstructor(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == 'optimization_SR':
        attacker = OptimizationSRAttacker(model, loss, cfg_attack, setup)        
    elif cfg_attack.attack_type == 'invertinggradients_diffusion':
        attacker = OptimizationDiffusionAttacker(model, loss, cfg_attack, setup)        
    elif cfg_attack.attack_type == 'optimization_group':
        attacker = OptimizationGroupAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == 'optimization_batch':
        attacker = OptimizationBasedAttackerBatch(model, loss, cfg_attack, setup)
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker


__all__ = ["prepare_attack"]
