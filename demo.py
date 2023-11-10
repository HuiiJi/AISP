from infer_model.inference import Inferencer
from train_model.make_dataloader import Testset
from pathlib import Path
import os
from utils.tools import simple_isp
import torch
import cv2



if __name__ == "__main__":
    import netron
    netron.start('/mnt/code/AISP_NR/infer_model/onnx/Unet_simplify.onnx')
    root_path = Path(os.path.abspath(__file__)).parent
    infer_config = root_path / 'infer_model' / 'infer_config.yaml'
    infer_path = root_path / 'output'
 
    test_loader = torch.utils.data.DataLoader(Testset(),
                                        batch_size=1,
                                        num_workers=0,
                                        pin_memory=True,
                                        shuffle=True)
    inferencer = Inferencer(infer_config)
    device = torch.device('cpu') if inferencer.args['forward_engine'] == 'qtorch' else torch.device('cuda')

    for step, images, in enumerate(test_loader):
        inputs = images.to(device)
        outputs = inferencer(inputs).clip(0, 1)
        output_rgb = simple_isp(outputs)
        input_rgb = simple_isp(inputs)
        cv2.imwrite(str(infer_path / f'{step}_output.png'), output_rgb)
        cv2.imwrite(str(infer_path / f'{step}_input.png'), input_rgb)
    
    
    