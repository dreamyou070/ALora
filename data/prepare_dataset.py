import os
from data.mvtec import MVTecDRAEMTrainDataset
import torch


def call_dataset(args) :
    # [1] set root data
    if args.do_object_detection :
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train_object_detector')
    else:
        root_dir = os.path.join(args.data_path, f'{args.obj_name}/train')
    data_class = MVTecDRAEMTrainDataset

    if args.answer_test:
        from data.mvtec import MVTecAnswerTrainDataset
        data_class = MVTecAnswerTrainDataset

        root_dir = os.path.join(args.data_path, f'{args.obj_name}/test')

        if args.trigger_word == 'teeth' :
            root_dir = os.path.join(args.data_path, f'{args.obj_name}/train') # training data #################################

    if args.trigger_word == 'transistor' :
        from data.mvtec_transistor import MVTecDRAEMTrainDataset as DatasetTransistor
        data_class = DatasetTransistor

    # ../../../MyData/anomaly_detection/Tuft/teeth_crop_onlyanomal/train

    print(f'data_class = {data_class}')
    print(f'root_dir = {root_dir}')

    tokenizer = None
    if not args.on_desktop :
        from model.tokenizer import load_tokenizer
        tokenizer = load_tokenizer(args)

    dataset = data_class(root_dir=root_dir,
                         anomaly_source_path=args.anomal_source_path,
                         resize_shape=[512, 512],
                         tokenizer=tokenizer,
                         caption=args.trigger_word,
                         use_perlin=True,
                         anomal_only_on_object=args.anomal_only_on_object,
                         anomal_training=True,
                         latent_res=args.latent_res,
                         do_anomal_sample =args.do_anomal_sample,
                         use_object_mask = args.do_object_detection,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    return dataloader

