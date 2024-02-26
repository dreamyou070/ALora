import os
from model.tokenizer import load_tokenizer
from data.mvtec import MVTecDRAEMTrainDataset
from data.mvtec_cropping import MVTecDRAEMTrainDataset_Cropping
import torch


def call_dataset(args) :

    tokenizer = load_tokenizer(args)

    # [1] set root data
    if args.do_object_detection :
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train_object_detector')
    else:
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train')
    data_class = MVTecDRAEMTrainDataset
    dataset = data_class(root_dir=root_dir,
                         anomaly_source_path=args.anomal_source_path,
                         resize_shape=[512, 512],
                         tokenizer=tokenizer,
                         caption=args.trigger_word,
                         use_perlin=True,
                         anomal_only_on_object=args.anomal_only_on_object,
                         anomal_training=True,
                         latent_res=args.latent_res,
                         do_anomal_sample =args.do_anomal_sample,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

