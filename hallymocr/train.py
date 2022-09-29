# -*- coding: utf-8 -*-

import sys
import time
import re

import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model

from nltk.metrics.distance import edit_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt['batch_max_length'])

        start_time = time.time()
        if 'CTC' in opt['Prediction']:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if opt['baiduCTC']:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt['baiduCTC']:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt['Prediction']:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt['sensitive'] and opt['data_filtering_off']:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

def train(opt):
    """ dataset preparation """
    if not opt['data_filtering_off']:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt['select_data'] = opt['select_data'].split('-')
    opt['batch_ratio'] = opt['batch_ratio'].split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open('./saved_models/{exp_name}/log_dataset.txt'.format(**opt), 'a')
    AlignCollate_valid = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=opt['PAD'])
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt['valid_data'], opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt['batch_size'],
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt['workers']),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    """ model configuration """
    if 'CTC' in opt['Prediction']:
        if opt['baiduCTC']:
            converter = CTCLabelConverterForBaiduWarpctc(opt['character'])
        else:
            converter = CTCLabelConverter(opt['character'])
    else:
        converter = AttnLabelConverter(opt['character'])
    opt['num_class'] = len(converter.character)

    if opt['rgb']:
        opt['input_channel'] = 3
    model = Model(opt)
    print('model input parameters', opt['imgH'], opt['imgW'], opt['num_fiducial'], opt['input_channel'],
          opt['output_channel'],
          opt['hidden_size'], opt['num_class'], opt['batch_max_length'], opt['Transformation'],
          opt['FeatureExtraction'],
          opt['SequenceModeling'], opt['Prediction'])

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt['saved_model'] != '':
        print('loading pretrained model from {}'.format(opt['saved_model']))
        if opt['FT']:
            model.load_state_dict(torch.load(opt['saved_model']), strict=False)
        else:
            model.load_state_dict(torch.load(opt['saved_model']))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt['Prediction']:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt['adam']:
        optimizer = optim.Adam(filtered_parameters, lr=opt['lr'])
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt['lr'], rho=opt['rho'], eps=opt['eps'])
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open('./saved_models/{}/opt.txt'.format(opt['exp_name']), 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        for k, v in opt.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        # print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt['saved_model'] != '':
        try:
            start_iter = int(opt['saved_model'].split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    num_iter = opt['num_iter']
    with tqdm(total = num_iter) as pbar:
        while (True):
            # train part
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt['batch_max_length'])
            batch_size = image.size(0)
            if 'CTC' in opt['Prediction']:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt['baiduCTC']:
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                    cost = criterion(preds, text, preds_size, length) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt['grad_clip'])  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            # validation part
            if (iteration + 1) % opt[
                'valInterval'] == 0 or iteration == 0:  # To see training progress, we also conduct validation when 'iteration == 0'
                elapsed_time = time.time() - start_time
                # for log
                with open('./saved_models/{}/log_train.txt'.format(opt['exp_name']), 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                            model, criterion, valid_loader, converter, opt)
                    model.train()

                    # training loss and validation loss
                    loss_log = f'[{iteration + 1}/{num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), './saved_models/{}/best_accuracy.pth'.format(opt['exp_name']))
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), './saved_models/{}/best_norm_ED.pth'.format(opt['exp_name']))
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')

                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                        if 'Attn' in opt['Prediction']:
                            gt = gt[:gt.find('[s]')]
                            pred = pred[:pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')

            # save model per 1e+5 iter.
            if (iteration + 1) % 1e+5 == 0:
                torch.save(
                    model.state_dict(), './saved_models/{}/iter_{}.pth'.format(opt['exp_name'], iteration + 1))

            if (iteration + 1) == opt['num_iter']:
                print('end the training')
                break

            iteration += 1
            pbar.update(1)
