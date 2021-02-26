import logging
import os
import os.path
from collections import deque
import itertools
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from PIL import Image
from torch.autograd import Variable

from cycada.data.adda_datasets import AddaDataLoader
from cycada.models import get_model
from cycada.models.models import models
from cycada.models import VGG16_FCN8s, Discriminator
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable

def check_label(label, num_cls):
    "Check that no labels are out of range"
    label_classes = np.unique(label.numpy().flatten())
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True



def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)
        
    return score, dis_score

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss
   
def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val)**2)
    else:
        _,_,h,w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1,h,w),requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
            preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--lr', '-l', default=0.0001)
@click.option('--momentum', '-m', default=0.9)
@click.option('--batch', default=1)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--crop_size', default=None, type=int)
@click.option('--half_crop', default=None)
@click.option('--cls_weights', type=click.Path(exists=True))
@click.option('--weights_discrim', type=click.Path(exists=True))
@click.option('--weights_init', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--lsgan/--no_lsgan', default=False)
@click.option('--num_cls', type=int, default=19)
@click.option('--gpu', default='0')
@click.option('--max_iter', default=10000)
@click.option('--lambda_d', default=1.0)
@click.option('--lambda_g', default=1.0)
@click.option('--train_discrim_only', default=False)
@click.option('--discrim_feat/--discrim_score', default=False)
@click.option('--weights_shared/--weights_unshared', default=False)


def main(output, dataset, datadir, lr, momentum, snapshot, downscale, cls_weights, gpu, 
        weights_init, num_cls, lsgan, max_iter, lambda_d, lambda_g,
        train_discrim_only, weights_discrim, crop_size, weights_shared,
        discrim_feat, half_crop, batch, model):
    
    # So data is sampled in consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    logdir = 'runs/{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(model, dataset[0],
            dataset[1], lr, lambda_d, lambda_g)
    if weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
    logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    print('Train Discrim Only', train_discrim_only)
    net = get_model(model, num_cls=num_cls, pretrained=True, weights_init=weights_init,
            output_last_ft=discrim_feat)
    if weights_shared:
        net_src = net # shared weights
    else:
        net_src = get_model(model, num_cls=num_cls, pretrained=True, 
                weights_init=weights_init, output_last_ft=discrim_feat)
        net_src.eval()

    odim = 1 if lsgan else 2
    idim = num_cls if not discrim_feat else 4096
    print('discrim_feat', discrim_feat, idim)
    print('discriminator init weights: ', weights_discrim)
    discriminator = Discriminator(input_dim=idim, output_dim=odim, 
            pretrained=not (weights_discrim==None), 
            weights_init=weights_discrim).cuda()

    loader = AddaDataLoader(net.transform, dataset, datadir, downscale, 
            crop_size=crop_size, half_crop=half_crop,
            batch_size=batch, shuffle=True, num_workers=2)
    print('dataset', dataset)

    # Class weighted loss?
    if cls_weights is not None:
        weights = np.loadtxt(cls_weights)
    else:
        weights = None
  
    # setup optimizers
    opt_dis = torch.optim.SGD(discriminator.parameters(), lr=lr, 
            momentum=momentum, weight_decay=0.0005)
    opt_rep = torch.optim.SGD(net.parameters(), lr=lr, 
            momentum=momentum, weight_decay=0.0005)

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    losses_dis = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    intersections = np.zeros([100,num_cls])
    unions = np.zeros([100, num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', max_iter)
   
    net.train()
    discriminator.train()

    while iteration < max_iter:
        
        for im_s, im_t, label_s, label_t in loader:
            
            if iteration > max_iter:
                break
           
            info_str = 'Iteration {}: '.format(iteration)
            
            if not check_label(label_s, num_cls):
                continue
            
            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            im_t = make_variable(im_t, requires_grad=False)
            label_t = make_variable(label_t, requires_grad=False)
           
            #############################
            # 2. Optimize Discriminator #
            #############################
            
            # zero gradients for optimizer
            opt_dis.zero_grad()
            opt_rep.zero_grad()
            
            # extract features
            if discrim_feat:
                score_s, feat_s = net_src(im_s)
                score_s = Variable(score_s.data, requires_grad=False)
                f_s = Variable(feat_s.data, requires_grad=False)
            else:
                score_s = Variable(net_src(im_s).data, requires_grad=False)
                f_s = score_s
            dis_score_s = discriminator(f_s)
            
            if discrim_feat:
                score_t, feat_t = net(im_t)
                score_t = Variable(score_t.data, requires_grad=False)
                f_t = Variable(feat_t.data, requires_grad=False)
            else:
                score_t = Variable(net(im_t).data, requires_grad=False)
                f_t = score_t
            dis_score_t = discriminator(f_t)
            
            dis_pred_concat = torch.cat((dis_score_s, dis_score_t))

            # prepare real and fake labels
            batch_t,_,h,w = dis_score_t.size()
            batch_s,_,_,_ = dis_score_s.size()
            dis_label_concat = make_variable(
                    torch.cat(
                        [torch.ones(batch_s,h,w).long(), 
                        torch.zeros(batch_t,h,w).long()]
                        ), requires_grad=False)

            # compute loss for discriminator
            loss_dis = supervised_loss(dis_pred_concat, dis_label_concat)
            (lambda_d * loss_dis).backward()
            losses_dis.append(loss_dis.item())

            # optimize discriminator
            opt_dis.step()

            # compute discriminator acc
            pred_dis = torch.squeeze(dis_pred_concat.max(1)[1])
            dom_acc = (pred_dis == dis_label_concat).float().mean().item() 
            accuracies_dom.append(dom_acc * 100.)

            # add discriminator info to log
            info_str += " domacc:{:0.1f}  D:{:.3f}".format(np.mean(accuracies_dom), 
                    np.mean(losses_dis))
            writer.add_scalar('loss/discriminator', np.mean(losses_dis), iteration)
            writer.add_scalar('acc/discriminator', np.mean(accuracies_dom), iteration)

            ###########################
            # Optimize Target Network #
            ###########################
           
            dom_acc_thresh = 60

            if not train_discrim_only and np.mean(accuracies_dom) > dom_acc_thresh:
              
                last_update_g = iteration
                num_update_g += 1 
                if num_update_g % 1 == 0:
                    print('Updating G with adversarial loss ({:d} times)'.format(num_update_g))

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if discrim_feat:
                    score_t, feat_t = net(im_t)
                    score_t = Variable(score_t.data, requires_grad=False)
                    f_t = feat_t 
                else:
                    score_t = net(im_t)
                    f_t = score_t

                #score_t = net(im_t)
                dis_score_t = discriminator(f_t)

                # create fake label
                batch,_,h,w = dis_score_t.size()
                target_dom_fake_t = make_variable(torch.ones(batch,h,w).long(), 
                        requires_grad=False)

                # compute loss for target net
                loss_gan_t = supervised_loss(dis_score_t, target_dom_fake_t)
                (lambda_g * loss_gan_t).backward()
                losses_rep.append(loss_gan_t.item())
                writer.add_scalar('loss/generator', np.mean(losses_rep), iteration)
                
                # optimize target net
                opt_rep.step()

                # log net update info
                info_str += ' G:{:.3f}'.format(np.mean(losses_rep))
               
            if (not train_discrim_only) and weights_shared and (np.mean(accuracies_dom) > dom_acc_thresh):
               
                print('Updating G using source supervised loss.')

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if discrim_feat:
                    score_s, _ = net(im_s)
                else:
                    score_s = net(im_s)

                loss_supervised_s = supervised_loss(score_s, label_s, 
                        weights=weights)
                loss_supervised_s.backward()
                losses_super_s.append(loss_supervised_s.item())
                info_str += ' clsS:{:.2f}'.format(np.mean(losses_super_s))
                writer.add_scalar('loss/supervised/source', np.mean(losses_super_s), iteration)

                # optimize target net
                opt_rep.step()

            # compute supervised losses for target -- monitoring only!!!
            loss_supervised_t = supervised_loss(score_t, label_t, weights=weights)
            losses_super_t.append(loss_supervised_t.item())
            info_str += ' clsT:{:.2f}'.format(np.mean(losses_super_t))
            writer.add_scalar('loss/supervised/target', np.mean(losses_super_t), iteration)

            ###########################
            # Log and compute metrics #
            ###########################
            if iteration % 10 == 0 and iteration > 0:
                
                # compute metrics
                intersection,union,acc = seg_accuracy(score_t, label_t.data, num_cls) 
                intersections = np.vstack([intersections[1:,:], intersection[np.newaxis,:]])
                unions = np.vstack([unions[1:,:], union[np.newaxis,:]]) 
                accuracy.append(acc.item() * 100)
                acc = np.mean(accuracy)
                mIoU =  np.mean(np.maximum(intersections, 1) / np.maximum(unions, 1)) * 100
              
                info_str += ' acc:{:0.2f}  mIoU:{:0.2f}'.format(acc, mIoU)
                writer.add_scalar('metrics/acc', np.mean(accuracy), iteration)
                writer.add_scalar('metrics/mIoU', np.mean(mIoU), iteration)
                logging.info(info_str)
                  
            iteration += 1

            ################
            # Save outputs #
            ################

            # every 500 iters save current model
            if iteration % 500 == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-itercurr.pth'.format(output))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-itercurr.pth'.format(output))

            # save labeled snapshots
            if iteration % snapshot == 0:
                os.makedirs(output, exist_ok=True)
                if not train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-iter{}.pth'.format(output, iteration))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-iter{}.pth'.format(output, iteration))

            if iteration - last_update_g >= len(loader):
                print('No suitable discriminator found -- returning.')
                torch.save(net.state_dict(), 
                        '{}/net-iter{}.pth'.format(output, iteration))
                iteration = max_iter # make sure outside loop breaks
                break

    writer.close()


if __name__ == '__main__':
    main()
