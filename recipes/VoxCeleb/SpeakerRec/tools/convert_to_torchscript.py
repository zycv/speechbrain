# -*- coding: utf-8 -*-
# Author: zouy68@gmail.com(ZY)

import argparse
import logging
import os
import torch

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.models.Xvector import Xvector


def convert(args):
    if args.model_type == 'TDNN':
        model = Xvector(in_channels=24)
    elif args.model_type == 'ECAPA':
        model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
        )
    else:
        logging.error("Model not supported! ", args.model_type)
        return
    model.load_state_dict(torch.load(args.raw_ckpt))
    scripted = torch.jit.script(model)
    scripted.save(args.target_pt)
    quantized_model = torch.quantization.quantize_dynamic(model,
                                                          {torch.nn.Linear},
                                                          dtype=torch.qint8)
    script_quant_model = torch.jit.script(quantized_model)
    script_quant_model.save(args.target_pt + '.quant')
    logging.info("Convert scripted and quantized scripted model success!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_ckpt',
                        type=str,
                        help='Path to raw embedding model to convert.')
    parser.add_argument('--target_pt',
                        type=str,
                        default='',
                        help='target pt file')
    parser.add_argument('--model_type',
                        default='TDNN',
                        type=str,
                        help="ECAPA or TDNN")
    args = parser.parse_args()
    if not os.path.exists(args.raw_ckpt):
        logging.error("File not exists: ", args.raw_ckpt)
        return
    if not args.target_pt:
        args.target_pt = args.raw_ckpt + '.pt'
    convert(args)


if __name__ == '__main__':
    main()
