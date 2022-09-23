# -*- coding: utf-8 -*-

import sys

import torch
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_validation(model, evaluation_loader, converter, opt):
    """ validation or evaluation """

    result = []

    for i, (image_tensors, path) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)

        preds_str = converter.decode(preds_index, length_for_pred)

        for s in preds_str:
            result.append(s.replace('[s]', ''))

    return result


def test(opt):
    """ model configuration """
    if 'CTC' in opt['Prediction']:
        converter = CTCLabelConverter(opt['character'])
    else:
        converter = AttnLabelConverter(opt['character'])
    opt['num_class'] = len(converter.character)

    if opt['rgb']:
        opt['input_channel'] = 3
    model = Model(opt)
    print('model input parameters', opt['imgH'], opt['imgW'], opt['num_fiducial'], opt['input_channel'], opt['output_channel'],
          opt['hidden_size'], opt['num_class'], opt['batch_max_length'], opt['Transformation'], opt['FeatureExtraction'],
          opt['SequenceModeling'], opt['Prediction'])
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt['saved_model'])
    model.load_state_dict(torch.load(opt['saved_model'], map_location=device))
    opt['exp_name'] = '_'.join(opt['saved_model'].split('/')[1:])
    # print(model)

    """ evaluation """
    model.eval()
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=opt['PAD'])
        eval_data = RawDataset(root=opt['test_data'], opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=int(opt['workers']),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        result = test_validation(model, evaluation_loader, converter, opt)

    return result


