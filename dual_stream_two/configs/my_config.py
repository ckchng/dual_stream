from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        # self.dataset = 'custom'
        self.dataset = 'customdualmask'
        self.data_root = '/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_32_len_200/'
        
        self.num_class = 1
        self.mean = [0.39509313, 0.39509313, 0.39509313]
        self.std = [0.17064099, 0.17064099, 0.17064099]
    
            
        # Model
        # self.model = 'bisenetv2'
        # self.model = 'bisenetv2dualmaskguided'
        self.model = 'bisenetv2dualmaskguidedv2'
        # self.model = 'bisenetv2dual'
        # self.model = 'bisenetv2dualht'
        # self.model = 'bisenetv2dualhtlastlayer'
        # self.model = 'litehrnet'

        # Training
        self.total_epoch = 30
        self.train_bs = 100
        self.loss_type = 'ohem_bce'
        # self.loss_type = 'bce'
        # self.lambda_s2 = 1.0
        
        # self.loss_type = 'dice_focal'
        # self.dfl_alpha = 0.65
        # self.dfl_gamma = 3.0
        
        self.dfl_pos_weight = 5.0  # foreground up-weight; tune based on fg/bg ratio
        
        # self.loss_type = 'ce'
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        # for litehrnet
        self.use_aux = True
        
        self.base_lr = 1e-2
        self.num_workers = 16
        # self.save_dir='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2_dualht_wo_pad2max/'
        # self.save_dir='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2/'
        # self.save_dir='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_df_bce_init/'
        self.save_dir='/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/snr_1_32_len_200/single_class/both_run1/'

        # Validating
        self.val_bs = 100   

        # Testing
        self.test_bs = 256
        # self.test_data_folder = '/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_15_longer_wo_blobs/images/train/'
        # self.test_data_folder2 = '/home/ckchng/Documents/SDA_ODA/LMA_data/snr_1_15_longer_wo_blobs/actual_images/train/'
        
        # self.load_ckpt_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/best.pth'
        # self.load_ckpt_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_snr_1_25_longer_wider/best.pth'
        # self.load_ckpt_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_ft_1_25/last.pth'
        self.save_mask = True
        # self.save_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bisenetv2_2/synthdata_w_blobs_result/vis/'
        # self.save_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_single_ch/vis/'

        # Training setting
        self.use_ema = False

        # Augmentation
        # self.crop_size = 304
        # self.crop_size = 192
        self.crop_size = None
        self.randscale = 0
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        # self.h_flip = 0.5

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/path/to/your/teacher/checkpoint'
        self.teacher_model = 'smp'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'


class MyConfig_a6000(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'customdualmask'
        # self.dataset = 'custom'
        # self.data_root = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_272_mask/'
        # self.data_root = '/home/ckchng/Documents/SDA_ODA/LMA_data/testing_gray_rt_272_mask/'
        # self.data_root = '/home/ckchng/Desktop/rt_seg/data/gray_rt_288_50_bg_for_dht/'
        # self.data_root = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_272_75_bg_for_dht/'
        # self.data_root = '/home/ckchng/Desktop/dataset/gray_rt_288_50_bg_for_dht/streak_only/'
        
        self.num_class = 1

        # Model
        # self.model = 'bisenetv2'
        self.model = 'bisenetv2dualmaskguidedv2'
        # self.model = 'bisenetv2dual'
        # self.model = 'bisenetv2dualht'
        # self.model = 'bisenetv2dualhtlastlayer'
        # self.model = 'litehrnet'

       # Training
        self.total_epoch = 50
        self.train_bs = 100
        self.loss_type = 'ohem_bce'
        self.lambda_s2 = 1.0
        # self.loss_type = 'dice_focal'
        self.dfl_alpha = 0.65
        self.dfl_gamma = 3.0
        

        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        # for litehrnet
        self.use_aux = True
        # self.use_aux = False
        self.base_lr = 1e-2
        self.num_workers = 16
        self.save_dir='/home/ckchng/Desktop/rt_seg/save/bg_50_no_crop/bisenetv2dualmaskguidedv2_singleCh_w_ohem_bce/'

        # Validating
        self.val_bs = 256   

        # Testing
        self.test_bs = 256
        self.test_data_folder = '/home/ckchng/Documents/SDA_ODA/LMA_data/gray_rt_272_mask/images/val/rt_272/'
        # self.load_ckpt_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bisenetv2/best.pth'
        self.save_mask = True
        # self.save_dir = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bisenetv2/50_bg/vis_on_test_set_1/'

        # Training setting
        self.use_ema = False

        # Augmentation
        # self.crop_size = 288
        # self.crop_size = 192
        self.crop_size = None
        self.randscale = 0
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.0

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/path/to/your/teacher/checkpoint'
        self.teacher_model = 'smp'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'
