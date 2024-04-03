import argparse, os, sys, datetime, glob
import time
from packaging import version
from omegaconf import OmegaConf
from transformers import logging as transf_logging

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
# import pytorch_lightning.strategies.deepspeed


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import check_config_attribute, get_empty_params_comparedwith_sd
from utils_train import set_logger, init_workspace, load_checkpoints, get_autoresume_path

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--ckpt_logdir", "-cl", type=str, default=None, help="directory for logging checkpoint")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    try:
        local_rank = int(os.environ.get('LOCAL_RANK'))
        global_rank = int(os.environ.get('RANK'))
        num_rank = int(os.environ.get('WORLD_SIZE'))
        #print(f'local_rank: {local_rank} | global_rank:{global_rank} | num_rank:{num_rank}')
    except:
        local_rank, global_rank, num_rank = 0, 0, 1
    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) 

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank, args.ckpt_logdir)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  


     ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    ckpt_workdir=workdir
    if args.ckpt_logdir is not None:
        ckpt_workdir = os.path.join(args.ckpt_logdir, args.name)
    config.model.params.logdir = ckpt_workdir
    model = instantiate_from_config(config.model)
    if args.auto_resume:
        ## the saved checkpoint must be: full-info checkpoint
        if getattr(config.model,'resume_path', None):
            resume_ckpt_path = config.model.resume_path
        else:
            resume_ckpt_path = get_autoresume_path(ckpt_workdir)
        if resume_ckpt_path is not None:
            args.resume_from_checkpoint = resume_ckpt_path
            logger.info("Resuming from checkpoint: %s"%args.resume_from_checkpoint)

            ## just in case train empy parameters only
            if check_config_attribute(config.model.params, 'empty_params_only') and check_config_attribute(config.model, 'sd_checkpoint'):
                _, model.empty_paras = get_empty_params_comparedwith_sd(model, config.model)
        else:
            model = load_checkpoints(model, config.model)
            logger.warning("Auto-resuming skipped as No checkpoit found!")
    else:
        model = load_checkpoints(model, config.model)
        
    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
        
    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr



    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False

    ## trainer config: others
    if "train" in config.data.params and config.data.params.train.target == "mira.data.hdvila.HDVila" or \
        ("validation" in config.data.params and config.data.params.validation.target == "mira.data.hdvila.HDVila"):
        trainer_kwargs['replace_sampler_ddp'] = False


    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")
    if args.train:
        if type(strategy_cfg) is str:
            if strategy_cfg == 'ddp':
                print('trainer is ddp!')
            # deepspeed
            elif strategy_cfg.startswith('deepspeed') :
                print('trainer is deepspeed!!!')
                model.enable_deepspeed = True
            else:
                print('trainer is not deepspeed!  ', strategy_cfg)
        else:
            print('trainer is ', strategy_cfg)
            if 'DeepSpeed' in strategy_cfg['target']:
                print('trainer is deepspeed!!!')
                model.enable_deepspeed = True

        if trainer_kwargs['precision'] == 32:
            trainer.fit(model, data)
        else:
            with torch.cuda.amp.autocast():
                trainer.fit(model, data)
