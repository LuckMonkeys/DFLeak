"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

from audioop import avgpp
import imp
from re import I
from tracemalloc import start
import torch
import time
import numpy as np
import torchvision
import math
from hydra.core.hydra_config import HydraConfig

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup
from PIL import Image, ImageChops
from breaching.analysis.metrics import psnr_compute

import logging

log = logging.getLogger(__name__)
import os


def doNoThing(img_lq, **kwargs):
    return img_lq

from breaching.attacks.utils import save_img, save_img_d,  get_mask, get_split_list


class OptimizationBasedAttacker(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        # except AttributeError:
        except Exception as e:
            print(e)
            pass  # No regularizers selected.
        # breakpoint()
        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.
        
        if self.cfg.model_sr.model_type is not None:
            if self.cfg.model_sr.model_type == "diface":
                from .bfr import diface_sr
                self.model_sr = diface_sr
            elif self.cfg.model_sr.model_type == "codeformer":
                from .bfr import codeformer_sr
                self.model_sr = codeformer_sr
            elif self.cfg.model_sr.model_type == "restoreformer":
                from .bfr import restoreformer_sr
                self.model_sr = restoreformer_sr
            elif self.cfg.model_sr.model_type == "guideDiffusion":
                from .accelarate.guided_diffusion import gdiffusion_sr
                self.model_sr = gdiffusion_sr
            elif self.cfg.model_sr.model_type == "doNoThing":
                self.model_sr = doNoThing

            self.percent_list = []
            self.model_sr_iterations = []
        self.callback = cfg_attack.optim.callback

    def __repr__(self):
        n = "\n"
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Hyperparameter Template: {self.cfg.type}

    Objective: {repr(self.objective)}
    Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False, activation=None, constraints=None, **kwargs):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        if activation is not None:
            labels = activation

        #save true user data
        if self.cfg.save.out_dir is not None and self.cfg.save.idx is None:
            if (true_user_data:=kwargs.get("true_user_data")) is not None:
                # save.out_dir / trail(user_idx) / iteration_{i}
                save_img(self.cfg.save.out_dir, true_user_data["data"].clone().detach(), iteration=-1, trial=f"user{kwargs['user_idx']}", dm=self.dm, ds=self.ds)

        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun, constraints, **kwargs)
                ]
                scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False, constraints=None, **kwargs):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels, constraints)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape], **kwargs)
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)


        rec_best_candidate = candidate.detach().clone()
        true_best_candidate = candidate.detach().clone()

        rec_best_candidate_idx = 0
        true_best_candidate_idx = 0

        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
        max_psnr_so_far = float("-inf")

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        
        current_wallclock = time.time()
        
        loss_history = []

        if (dir_path:=self.cfg.save.out_dir) is not None:
            log.setLevel(logging.INFO)
            if not os.path.exists( dir_path):
                os.makedirs(dir_path, exist_ok=True)
            log.addHandler(logging.FileHandler(dir_path + '/log'))
        try:
            for iteration in range(self.cfg.optim.max_iterations):

                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration, **kwargs)
                time_closure_prefix = time.time() - current_wallclock
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                time_update = time.time() - current_wallclock - time_closure_prefix
                scheduler.step()
                
                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)


                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        rec_best_candidate = candidate.detach().clone()
                        rec_best_candidate_idx = iteration
                    
                    
                    if self.cfg.optim.patched is not None and self.cfg.optim.patched > 1:
                        patch_size = int(self.cfg.optim.patched)
                        avg_pool = torch.nn.AvgPool2d(patch_size, patch_size)
                        mean_candidate = avg_pool(candidate)
                        reshpae_candidate = mean_candidate.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
                        candidate.copy_(reshpae_candidate)
                    
                
                with torch.no_grad():

                    if self.cfg.model_sr.model_type is not None and iteration in self.model_sr_iterations:
                        
                        interval_index =  self.model_sr_iterations.index(iteration)
                        
                        if self.cfg.save.out_dir is not None and self.cfg.save.save_hq:
                            path = os.path.join(self.cfg.save.out_dir, f"user{kwargs['user_idx']}", f"{iteration}_0.png")
                            save_img_d(path, candidate.clone().detach(), dm=self.dm, ds=self.ds)
                            
                        img_lq = torch.clamp(rec_best_candidate.clone().detach() * self.ds + self.dm, 0, 1)
                        img_lq = torch.repeat_interleave(img_lq, self.cfg.model_sr.repeat, dim=0) # repeat multiple times
                        img_hq_repeat = self.model_sr(img_lq=img_lq, **kwargs) #[B*repeat, 3, h, w]   [0, 1]
                        
                        img_hq = torch.zeros_like(candidate) 
                        
                         

                        if self.cfg.model_sr.model_type is not None:
                            for i in range(candidate.shape[0]): 
                                img_hq[i] = torch.mean(img_hq_repeat[i*self.cfg.model_sr.repeat:(i+1)*self.cfg.model_sr.repeat], dim=0)
                            
                            mask_list = [] 
                            for i in range(candidate.shape[0]):
                                img_batch = torch.stack([img_hq[i].clone(), candidate[i].clone()*self.ds.squeeze(0) + self.dm.squeeze(0) ], dim=0) #shape [2, 3, h, w]

                                mask = get_mask(img_batch.cpu()*255, percent=self.percent_list[interval_index]) # [1, h, w]
                                mask = torchvision.transforms.ToTensor()(mask)

                                mask_list.append(mask)
                            mask = torch.stack(mask_list, dim=0).to(**self.setup) #shape [B, 1, h, w]
                            
                            
                            
                            if (prev_mask:=kwargs.get("mask")) is not None:
                                prev_inverse_mask = 1 - prev_mask
                                partial_mask = torch.add(mask, prev_inverse_mask) # only take the different area from previous mask
                                partial_mask[partial_mask == 2] = 1 #remove the 2 in the added mask
                                full_mask = torch.mul(mask, prev_mask)
                            else:
                                partial_mask = full_mask = mask
                                
                            log.info(f"current percent: {( torch.sum(full_mask == 0).item() / full_mask.numel()) * 100:.2f}")
                            inverse_mask = 1 - full_mask #

                            
                            #set grad mask
                            kwargs["mask"] = full_mask

                            weight_full_mask = full_mask.clone()
                            weight_inverse_mask = inverse_mask.clone()
                            
                            if self.cfg.model_sr.lq_weight is not None:
                                weight_full_mask[weight_full_mask == 1] -= self.cfg.model_sr.lq_weight 
                                inverse_mask[weight_inverse_mask == 0] +=  self.cfg.model_sr.lq_weight 

                            
                            #normalize
                            img_hq = (img_hq - self.dm) / self.ds
                            
                            # save hq image
                            if self.cfg.save.out_dir is not None and self.cfg.save.save_hq:
                                path = os.path.join(self.cfg.save.out_dir, f"user{kwargs['user_idx']}", f"{iteration}_1.png")
                                save_img_d(path, img_hq.clone().detach(), dm=self.dm, ds=self.ds)

                            candidate_area = torch.einsum('blhw,bchw->blhw', candidate.clone(), weight_full_mask)
                            hq_area = torch.einsum('blhw,bchw->blhw', img_hq.clone(), weight_inverse_mask)
                            
                            merge_img = candidate_area + hq_area
                            candidate.copy_(merge_img)
                            
                            #save merge image
                            if self.cfg.save.out_dir is not None and self.cfg.save.save_hq:
                                path = os.path.join(self.cfg.save.out_dir, f"user{kwargs['user_idx']}", f"{iteration}_2.png")
                                save_img_d(path, merge_img.clone().detach(), dm=self.dm, ds=self.ds)

                        if self.cfg.regularization.pix.scale > 0: 
                            kwargs["ref_candidate"] = img_hq.detach().clone()
                        
                        if self.cfg.model_sr.scale_tv is not None: 
                            kwargs["tv_scale_factor"] = self.cfg.model_sr.scale_tv

                        # reinit optimizer and scheduler
                        interval_iterations = 0 # the minimum interval iteration is 1
                        log.info(f"Use SR model to replace candidate, iteration: {iteration}, lr: {optimizer.param_groups[0]['lr']}, percent: {str(self.percent_list)}, interval_iterations: {interval_iterations}")
                        
                        # self.cfg.optim.callback = 100
                        self.callback = self.cfg.optim.callback / 10

                time_projectImage = time.time() - current_wallclock - time_update
                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.callback == 0:
                    
                    f_regularizer = ''
                    for regularizer_objective in self.current_regularizer_objectives:
                        f_regularizer += f'{regularizer_objective:2.4f} '

                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        f" regularizer_objective_loss: {f_regularizer} "
                        f"max psnr so far: {max_psnr_so_far:.2f}" 
                    )
                    current_wallclock = timestamp
                    if self.cfg.save.out_dir is not None:
                        if self.cfg.save.idx is not None:
                            path = self.cfg.save.out_dir + f'/lq_{self.cfg.optim.max_iterations}/{self.cfg.save.idx}.png' #only save the last iteration image
                            save_img_d(path, candidate.clone().detach(), dm=self.dm, ds=self.ds)

                        else: 
                            save_img(os.path.join(self.cfg.save.out_dir, str(trial)), candidate.clone().detach(), iteration=iteration, trial=f"user{kwargs['user_idx']}", dm=self.dm, ds=self.ds) # save.out_dir/ trial / user_idx / iteration_{i}

                if objective_value.item() < self.cfg.model_sr.model_type.loss_threshold:
                
                    self.model_sr_iterations = get_split_list(iteration+1, self.cfg.optim.max_iterations, self.cfg.model_sr.times, dtype=np.int32)
                    self.percent_list = get_split_list(self.cfg.model_sr.percent_start, self.cfg.model_sr.percent_end, self.cfg.model_sr.times, dtype=np.int32)
                
                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        log.info(f"reconstruction best candidate found in  {rec_best_candidate_idx} iteration."
                 f"true best candidate found in  {true_best_candidate_idx} iteration."
                 )
        self.callback = self.cfg.optim.callback
        
        if self.cfg.save.save_loss:
            loss_path = os.path.join(HydraConfig.get().runtime.output_dir, f"loss_user{kwargs['user_idx']}")
            np.savetxt(loss_path, np.array(loss_history), delimiter=",", fmt='%s')
        
        if self.cfg.save_true_best_candidate: 
            return true_best_candidate.detach()
        
        return rec_best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration, activation=None, **kwargs):
        def closure():

            start_time = time.time()

            optimizer.zero_grad()
            time_zero_grad = time.time() - start_time


            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            time_augmentation = time.time() - start_time - time_zero_grad

            total_objective = 0
            total_task_loss = 0
            
            if self.cfg.update_weights == "equal":
                # update_weights = [1 for i in range(rec_model)]
                update_weights = torch.ones(len(rec_model), dtype=candidate.dtype).to(candidate.device)
            elif self.cfg.update_weights == "exp":
                update_weights = torch.arange(len(rec_model), 0, -1, dtype=candidate.dtype, device=candidate.device)
                update_weights = update_weights.softmax(dim=0)
                # update_weights  = weights / weights[-1]
            elif self.cfg.update_weights == 'linear':
                update_weights = torch.arange(len(rec_model), 0, -1, dtype=candidate.dtype, device=candidate.device)
                update_weights = update_weights / torch.sum(update_weights)
            # breakpoint() # check the update_weights
            for update_id, (model, data) in enumerate(zip(rec_model, shared_data)): 
                if self.cfg.layer_weights == None:
                    layer_weights = None
                elif self.cfg.layer_weights == 'equal':
                    layer_weights = [1 for i in range(len(data["gradients"]))]
                elif self.cfg.layer_weights == 'AGIC':  #TODO need to varify
                    layer_weights = []
                    def get_layer_weights(N_conv, beta, gradients):
                        from statistics import mean
                        """
                            Default the last two layer is gradients of FC layer
                        """
                        weights = []
                        for i in range(1, N_conv+1):
                            #weight for conv layer
                            p = torch.count_nonzero(gradients[i-1]) / torch.numel(gradients[i-1])
                            weight = (1 + (i - 1)(beta - 1) / (N_conv - 1)) / (1 - p)
                            weights.append(weight)
                        #weight for FC layer
                        fc_weight = mean(weights)
                        weights.append(fc_weight)

                        return weights
                    layer_weights = get_layer_weights(N_conv=len(data["gradients"]), beta=2, gradients=data["gradients"])

                elif self.cfg.layer_weights.endswith('idx'):
                    layer_weights = torch.load(f'./breaching/attacks/auxiliaries/{self.cfg.layer_weights}').to(torch.int64)
                else:
                    raise NotImplementedError('Please input correct layer_weights')

                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels, layer_weights=layer_weights)
                total_objective += objective * update_weights[update_id]
                total_task_loss += task_loss

            time_loss1back = time.time() - start_time - time_zero_grad - time_augmentation
            total_objective /= len(rec_model)
            total_task_loss /= len(rec_model)

            regularizer_objectives = []
            for regularizer in self.regularizers:
                regularizer_objective = regularizer(candidate_augmented, **kwargs)
                regularizer_objectives.append(regularizer_objective)
                total_objective += regularizer_objective




            if total_objective.requires_grad:
                loss2back_start_time = time.time()
                total_objective.backward(inputs=candidate, create_graph=False)
                time_loss2back = time.time() - loss2back_start_time

            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))

                if (mask:=kwargs.get("mask")) is not None and self.cfg.model_sr.grad_mask_adjust is not None:
                    grad_mask = mask.clone()
                    grad_mask[grad_mask == 0] += self.cfg.model_sr.grad_mask_adjust # reduce the gradient of mask area
                    candidate.grad = torch.einsum('blhw,bchw->blhw', candidate.grad, grad_mask)


                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            self.current_regularizer_objectives = regularizer_objectives
            # self.aug_task_loss = aug_task_loss if self.cfg.aug_task_loss else None
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                layer_weights = [1 for i in range(len(data["gradients"]))]
                score += objective(model, data["gradients"], candidate, labels, layer_weights)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution selected in restart {optimal_index} with rec. loss {optimal_val.item():2.4f}")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)
