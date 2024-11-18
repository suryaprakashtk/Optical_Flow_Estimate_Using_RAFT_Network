import argparse
from torchsummary import summary
from collections import OrderedDict

from raft import *
import trainer
import dataloader
import os

def get_model_summary(model):
    print("Model Summary: \n")
    model_summary = summary(model, [(3, 384, 512), (3, 384, 512)])

def load_weights(model, path):
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
    
    if 'raft-small.pth' in path:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():

            if 'encoder.' in k:
                k = k.replace('encoder.', '')
            if 'convc1' in k:
                k = k.replace('convc1', 'corr.0')
            if 'convf1' in k:
                k = k.replace('convf1', 'flow.0')
            if 'convf2' in k:
                k = k.replace('convf2', 'flow.2')
            if 'gru.convz' in k:
                k = k.replace('gru.convz', 'convz')
            if 'gru.convr' in k:
                k = k.replace('gru.convr', 'convr')
            if 'gru.convq' in k:
                k = k.replace('gru.convq', 'convq')
            if 'flow_head' in k:
                k = k.replace('flow_head.', '')
            if 'module.update_block' in k:
                k = k.replace('module.update_block', 'update_block')

            if 'module.fnet' in k:
                k = k.replace('module.fnet', 'feature_encoder')
            if 'module.cnet' in k:
                k = k.replace('module.cnet', 'context_encoder')
            if 'layer' in k:
                k = k.replace('layer', 'res_unit')
            if 'downsample' in k:
                k = k.replace('downsample.0', 'conv4')

            new_state_dict[k] = v

        # load params
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print('Weights Loaded')

    model.eval()
    
    return model


def main(args):
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    dataset_path = "datasets/"
    
    if not os.path.exists(f'train_{args.name}'):
        os.makedirs(f'train_{args.name}')
    if not os.path.exists(f'train_{args.name}/images'):
        os.makedirs(f'train_{args.name}/images') 
    if not os.path.exists(f'train_{args.name}/plots'):
        os.makedirs(f'train_{args.name}/plots')
    
    if args.test:
        model = RAFT(device, args)
        model.to(device)
        path = f'train_{args.name}/{args.checkpoints}_model.pth'

        # Load checkpoint model wieghts
        model = load_weights(model, path)

        test_loader = dataloader.get_test_dataloader(dataset_path)
        model = trainer.testRAFT(model, test_loader, device, args)
        
        return model

    
    model = RAFT(device, args)
    model.to(device)
    
    # Print model summary
    get_model_summary(model)

    if args.checkpoints == 0:
        path = 'models/raft-small.pth'

        # Load pretrained small model wieghts
        model = load_weights(model, path)

    elif args.checkpoints != 0:
        path = f'train_{args.name}/{args.checkpoints}_model.pth'

        # Load checkpoint model wieghts
        model = load_weights(model, path)


    # set-up crowdflow dataloader
    print("Set up dataloader")
    train_loader = dataloader.get_train_dataloader(dataset_path)
    test_loader = dataloader.get_test_dataloader(dataset_path)


    # Perform dataset specific training
    model = trainer.trainRAFT(model, train_loader, device, args)
    
    # Test mdoel
    model = trainer.testRAFT(model, test_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-attention', help="name your experiment") # raft-basic, raft-attention, raft-charbonnier, rat-attention-charbonnier
    parser.add_argument('--test', default=False)
    parser.add_argument('--checkpoints', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--loss', default='L1_loss') # L1-loss, charbonnier-loss
    args = parser.parse_args()

    main(args)
