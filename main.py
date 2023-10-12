"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf
import torch

import datetime
import time
import logging

import breaching

import os
from hydra.core.hydra_config import HydraConfig

os.environ["HYDRA_FULL_ERROR"] = "0"
log = logging.getLogger(__name__)


def main_process(process_idx, local_group_size, cfg, num_trials=100):
    """This function controls the central routine."""
    total_time = time.time()  # Rough time measurements here
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)
    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)
    
    if cfg.state_dict_path is not None and not os.path.isdir(cfg.state_dict_path):
        extension = os.path.splitext(cfg.state_dict_path)[-1]
        if extension == '.pth':
            model.model.load_state_dict(torch.load(cfg.state_dict_path))
        elif extension == '.ckpt':
            model.load_state_dict(torch.load(cfg.state_dict_path)['state_dict'])

    if cfg.case.server.model_state == "multiple_updates":
        if not os.path.isdir(cfg.state_dict_path):
            raise Exception("Server model state is multiple_updates, please input a dir of state_dict_path!")


    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    if cfg.case.user.user_idx is not None:
        print("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")
    log.info(
        f"Partitioning is set to {cfg.case.data.partition}. Make sure there exist {num_trials} users in this scheme."
    )

    if (trials_choose:=cfg.get("trials_choose")) is not None:
        num_trials = len(trials_choose)

    if cfg.case.user.user_idx <= 0:
        cfg.case.user.user_idx = -1
    run = 0
    overall_metrics = []
    while run < num_trials:
        local_time = time.time()
        # Select data that has not been seen before:
        if trials_choose is not None:
            cfg.case.user.user_idx = trials_choose[run]
        else:
            cfg.case.user.user_idx += 1
        # breakpoint()
        try:
            user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
        except ValueError:
            log.info("Cannot find other valid users. Finishing benchmark.")
            break
        if cfg.case.data.modality == "text":
            dshape = user.dataloader.dataset[0]["input_ids"].shape
            data_shape_mismatch = any([d != d_ref for d, d_ref in zip(dshape, cfg.case.data.shape)])
        else:
            data_shape_mismatch = False  # Handled by preprocessing for images
        if len(user.dataloader.dataset) < user.num_data_points or data_shape_mismatch:
            log.info(f"Skipping user {user.user_idx} (has not enough data or data shape mismatch).")
        else:
            log.info(f"Now evaluating user {user.user_idx} in trial {run}.")
            run += 1
            # Run exchange
            shared_user_data, payloads, true_user_data = server.run_protocol(user, path=cfg.state_dict_path)
            # Evaluate attack:
            # try:
            if cfg.get("save_user_gradients") is not None:
                gradients = shared_user_data[0]["gradients"]
                torch.save(gradients, f"{cfg.attack.save.out_dir}/gradients_{user.user_idx}.pth")
                continue


            kwargs = {"cfg_data":cfg.case.data, "user_idx":user.user_idx}
            if cfg.attack.init == "real-specify":
                kwargs["img_base_dir"] = cfg.attack.img_base_dir
            
            kwargs["user_idx"] = user.user_idx

            attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
            reconstruction, stats = attacker.reconstruct(
                payloads, shared_user_data, server.secrets, dryrun=cfg.dryrun, **kwargs)

            # Run the full set of metrics:
            if cfg.save_local_summary:
                metrics = breaching.analysis.report(
                    reconstruction,
                    true_user_data,
                    payloads,
                    server.model,
                    order_batch=True,
                    compute_full_iip=True,
                    compute_rpsnr=True,
                    compute_ssim=True,
                    cfg_case=cfg.case,
                    setup=setup,
                )
                # Add query metrics
                metrics["queries"] = user.counted_queries

                # Save local summary:
                breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time, original_cwd=False, hydra_output_dir=HydraConfig.get().runtime.output_dir)
                overall_metrics.append(metrics)
            # Save recovered data:
            if cfg.save_reconstruction:
                breaching.utils.save_reconstruction(reconstruction, payloads, true_user_data, cfg, hydra_output_dir=HydraConfig.get().runtime.output_dir)
            if cfg.dryrun:
                break
            # except Exception as e:  # noqa # yeah we're that close to the deadlines
            #     log.info(f"Trial {run} broke down with error {e}.")

    # Compute average statistics:
    if cfg.save_global_summary:
        if len(overall_metrics) == 0:
            log.info("No valid trials were run. Skipping global summary.")
        else:
            average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

            # Save global summary:
            breaching.utils.save_summary(
                cfg, average_metrics, stats, time.time() - total_time, original_cwd=True, table_name="BENCHMARK_breach", hydra_output_dir=HydraConfig.get().runtime.output_dir)


@hydra.main(version_base="1.1", config_path="breaching/config", config_name="cfg")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = 233  # The benchmark seed is fixed by default!

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


if __name__ == "__main__":
    main_launcher()

