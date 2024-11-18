import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np

import flow_viz
import Read_Write_Files as rw_files
import matplotlib.pyplot as plt

def trainRAFT(model, train_loader, device, args):

    lr = 1e-4
    weight_decay = 2e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    epes = []
    train_losses = []
    train_epes= []

    print_every_epoch = 20
    n = args.checkpoints
    if n == 0:
        n = n-1
    for e in range(n+1, args.epochs):
        for i, data in tqdm(enumerate(train_loader)):
            model.train() 
            optimizer.zero_grad()
            image1, image2, flow_gt = [x for x in data]
            image1, image2, flow_gt = image1.to(device), image2.to(device), flow_gt.to(device)

            flow_predictions = model(image1, image2)  
            loss = compute_loss(flow_predictions, flow_gt, args)
            losses.append(loss.item())
            
            epe = compute_epe_metric(flow_predictions, flow_gt)
            epes.append(epe)
      
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if i == 0:
                print('Epoch %d, Iterations %d, loss = %.4f' % (e, i, loss.item()))

        train_losses.append(np.mean(np.array(losses)))
        losses = []
        train_epes.append(np.mean(np.array(epes)))
        epes = []

        if e % print_every_epoch == 0:
            PATH = f'train_{args.name}/%d_model.pth' % (e)
            torch.save(model.state_dict(), PATH)
            np.savez(f'train_{args.name}/train_metric.npz', epochs=np.array(np.arange(n+1, e)), train_loss=np.array(train_losses), epe=np.array(train_epes))

    PATH = f'train_{args.name}/%d_model.pth' % (e)
    torch.save(model.state_dict(), PATH)
    np.savez(f'train_{args.name}/train_metric.npz', epochs=np.array(np.arange(n+1, e)), train_loss=np.array(train_losses), epe=np.array(train_epes))

    return model

def compute_loss(flow_preds, flow_gt, args, gamma=0.8, max_flow=400):   
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (mag < max_flow)

    for i in range(len(flow_preds)):
        if 'L1' in args.loss:
            loss = (flow_preds[i] - flow_gt).abs()
        else:
            loss = charbonnier_loss(flow_preds[i], flow_gt)
            
        flow_loss += (gamma**(len(flow_preds) - i - 1)) * (valid[:, None] * loss).mean()

    return flow_loss

def charbonnier_loss(x, y, epsilon=1e-5):
    diff = x - y
    error = torch.sqrt(diff * diff + epsilon)
    loss = torch.mean(error)
    return loss

def compute_epe_metric(flow_preds, flow_gt, max_flow=400):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (mag < max_flow)

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    return epe.mean().item() 

test_losses = []
test_epes = []
def testRAFT(model, test_loader, device, args):
    losses = []
    epes= []

    for i, data in enumerate(test_loader):
        model.eval() 
        image1, image2, flow_gt = [x for x in data] 
        image1, image2, flow_gt = image1.to(device), image2.to(device), flow_gt.to(device)

        flow_predictions = model(image1, image2)

        loss = compute_loss(flow_predictions, flow_gt, args)
        test_losses.append(loss.item())

        epe = compute_epe_metric(flow_predictions, flow_gt)
        test_epes.append(epe)

        image1 = image1[0].permute(1,2,0).detach().cpu().numpy()

        flo = flow_predictions[-1][0].permute(1,2,0).detach().cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo)
        img_flo = np.concatenate([image1, flo], axis=0)

        # Visualize Flow
        plt.imshow(img_flo / 255.0)
        plt.show()
        plt.savefig(f'train_{args.name}/images/flow{args.checkpoints}_{i}.png')
              
    print(f'Testing: Loss: {test_losses}, EPE: {test_epes}')
    line = f'epochs: {args.checkpoints}, Test Loss: {test_losses}, Test EPE: {test_epes} \n'
    with open(f'train_{args.name}/test_results.txt', 'a') as f:
        f.write(line)
    return model