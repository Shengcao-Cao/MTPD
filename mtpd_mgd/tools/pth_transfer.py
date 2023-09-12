import torch
import argparse
from collections import OrderedDict

def change_model(args):
    mgd_model = torch.load(args.mgd_path, 'cpu')
    all_name = []
    for name, v in mgd_model["state_dict"].items():
        if name.startswith("student."):
            all_name.append((name[8:], v))
        else:
            continue
    state_dict = OrderedDict(all_name)
    del mgd_model['optimizer']
    del mgd_model['meta']
    mgd_model['state_dict'] = state_dict
    torch.save(mgd_model, args.output_path) 

           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--mgd_path', type=str, default='work_dirs/mgd_psp_r101_distill_deepv3_r18_40k_512x512_city/latest.pth', 
                        metavar='N',help='mgd_model path')
    parser.add_argument('--output_path', type=str, default='deeplabv3_res18_new.pth',metavar='N', 
                        help = 'output path')
    args = parser.parse_args()
    change_model(args)
