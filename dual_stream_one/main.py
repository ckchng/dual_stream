from core import SegTrainer, DualSegTrainer, DualMaskTrainer
from configs import MyConfig, load_parser, MyConfig_a6000

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = MyConfig()
    # config = MyConfig_a6000()

    # If you want to use command-line arguments, please uncomment the following line
    config = load_parser(config)

    config.init_dependent_config()

    if config.model == 'bisenetv2dual' or config.dataset == 'customdual':
        trainer = DualSegTrainer(config)
    elif config.model == 'bisenetv2dualmaskguided' or config.dataset == 'customdualmask':
        trainer = DualMaskTrainer(config)
    else:
        trainer = SegTrainer(config)

    if config.task == 'train':
        trainer.run(config)
    elif config.task == 'val':
        trainer.validate(config)
    elif config.task == 'predict':
        trainer.predict(config)
    else:    
        raise ValueError(f'Unsupported task type: {config.task}.\n')