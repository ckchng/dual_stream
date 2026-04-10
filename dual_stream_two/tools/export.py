import os, sys, torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from configs import MyConfig, load_parser
from models import get_model

import warnings
warnings.filterwarnings("ignore")


class Exporter:
    def __init__(self, config):
        config.use_aux = False
        config.use_detail_head = False

        self.load_ckpt_path = config.load_ckpt_path
        self.export_format = config.export_format
        self.export_size = config.export_size
        self.onnx_opset = config.onnx_opset
        self.export_path = config.export_name + f'.{config.export_format}'
        self.config = config

        self.model = get_model(config)
        self.load_ckpt()

    def load_ckpt(self):
        if not self.load_ckpt_path:     # when set to None
            pass
        elif os.path.isfile(self.load_ckpt_path):
            checkpoint = torch.load(self.load_ckpt_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.eval()

            print(f'Loading checkpoint: {self.load_ckpt_path} successfully.\n')
            del checkpoint
        else:
            raise RuntimeError

    def export(self):
        print('\n=========Export=========')
        print(f'Model: {self.config.model}\nEncoder: {self.config.encoder}\nDecoder: {self.config.decoder}')
        print(f'Export Size (H, W): {self.export_size}')
        print(f'Export Format: {self.export_format}')

        if self.export_format == 'onnx':
            from models.modules import replace_adaptive_avg_pool
            self.model = replace_adaptive_avg_pool(self.model)

            self.export_onnx()
            print(f'\nExporting Finished. Model saved to: {os.path.abspath(self.export_path)}\n')

        else:
            raise NotImplementedError

    def export_onnx(self, image=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        is_dual = hasattr(self.config, 'model') and 'dual' in self.config.model.lower()
        if is_dual:
            export_size2 = getattr(self.config, 'export_size2', self.export_size)
            x1 = torch.rand(1, 3, *self.export_size).to(device)
            x2 = torch.rand(1, 3, *export_size2).to(device)
            dummy_input = (x1, x2)
            input_names = ['input_stream1', 'input_stream2']
            output_names = ['output_main', 'output_stream2']
        else:
            dummy_input = torch.rand(1, 3, *self.export_size).to(device) if not image else image
            input_names = ['input']
            output_names = ['output']

        dynamic_axes = {
            'input_stream1': {0: 'batch'},
            'input_stream2': {0: 'batch'},
            'output_main':   {0: 'batch'},
            'output_stream2':{0: 'batch'},
        }
        torch.onnx.export(self.model, dummy_input, self.export_path, opset_version=self.onnx_opset,
                          input_names=input_names, output_names=output_names, dynamo=False, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    config = MyConfig()
    config = load_parser(config)
    config.model = 'bisenetv2dualmaskguidedv2'
    config.num_class = 2
    config.export_size = (192, 416)   # stream1: RT map size (H, W)
    # config.export_size = (192, 288)   # stream1: RT map size (H, W)
    config.export_size2 = (288, 288)  # stream2: tile size (H, W)
    config.load_ckpt_path = '/home/ckchng/Documents/realtime-semantic-segmentation-pytorch-main/save/bg_50_no_crop/gray_rt_288_snr_1_25_new_bg_longer_dimmer/best.pth'
    config.export_name = 'best'  # output filename (without .onnx)
    config.init_dependent_config()

    try:
        exporter = Exporter(config)
        exporter.export()
    except Exception as e:
        print(f'\nUnable to export PyTorch model {config.model} to {config.export_format} due to: {e}')