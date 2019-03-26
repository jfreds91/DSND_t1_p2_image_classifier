# The purpose of this program is to use an existing model
# to predict an image class

# expected arguments:
# img_filepath - the filepath for the desired image
# checkpoint - the filepath for the checkpoint.pth file to use

# optional arguments
# --top_k - the top k guesses to return
# --category_names - a json mapping of categories to real names
# -- gpu - use gpu inference

# imports
import argparse
import os
import sys
import torch
from collections import OrderedDict
import numpy as np
import PIL

torch.cuda.empty_cache()

###############################################################################
################################### Argparse ##################################
###############################################################################

parser = argparse.ArgumentParser(description='NN prediction python executable')

# nonoptional arguments
parser.add_argument('img_filepath', action='store', type=str,
                    help='The target image filepath.')
parser.add_argument('checkpoint', action='store', type=str,
                    help='The checkpoint.pth file to load as model.')

# optional arguments
parser.add_argument('--top_k', type=int, dest='user_top_k', default=3,
                    help='The number of best guesses to return. Default 3.')
parser.add_argument('--category_names', type=str, dest='category_names',
                    default=None,
                    help='The mapping of image names to categories. Must be \
                    .json file located in same folder as .py file.')
parser.add_argument('--gpu', action='store_true', dest = 'tryGPU',
                    help='Attempt to run on GPU. Default False.')

args = parser.parse_args()

# all above variables now accessible through args.variablename
# error check, special cases

if args.img_filepath == 'test':
    args.img_filepath = 'C:/Users/Jesse/Documents/PyData/DSND_Term1/projects/p2_image_classifier/flower_data.tar/valid/1/image_06739.jpg'
        
###############################################################################
################################ Model Inference ##############################
###############################################################################

dir_path = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

def loadmodel(path):
    checkpoint = torch.load(path)
    model_arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    #outputlen = checkpoint['outputlen']
    input_num = checkpoint['input_num']
    
    model_method = getattr(models, model_arch)
    model = model_method(pretrained=True)
    model.class_to_idx = checkpoint['label_idx']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_num, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, len(model.class_to_idx))),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    # scales, crops, and normalizes a PIL image for a PyTorch mode,
    # returns a tensor
    
    im = PIL.Image.open(image)
    
    # resize image
    width, height = im.size
    if height >= width:
        newsize = (256, int(height*256/width))
    else:
        newsize = (int(width*256/height), 256)
    im = im.resize(newsize)
    
    # centercut (left, upper, right, lower)
    width, height= im.size
    left = int((width - 224)/2)
    upper = int((height - 224)/2)
    right = int(left + 224)
    lower = int(upper + 224)
    im = im.crop((left, upper, right, lower))
    
    # convert to pytorch format
    np_image = np.array(im)/255 # normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image)


def predict(image_path, model, topk, tryGPU):
    # get tensor from image path
    
    tensor = process_image(image_path)

    # check if GPU
    if args.tryGPU:
        if torch.cuda.is_available():
            device = 'cuda:0'
            print('running on: ' + device)
            model.to(device)
            model.eval()
            # must be a cuda tensor
            logps = model.forward(torch.unsqueeze(tensor, 0).type(torch.cuda.FloatTensor))
            ps = torch.exp(logps)
            top_ps, top_classes = ps.cpu().topk(topk, 1)
        else:
            print('GPU unavailable. Try without GPU')
            sys.exit()
    else:
        device = 'cpu'
        print('running on: ' + device)
        model.to(device)
        model.eval()
        # must be a cpu tensor
        logps = model.forward(torch.unsqueeze(tensor, 0).type(torch.FloatTensor))
        ps = torch.exp(logps)
        top_ps, top_classes = ps.topk(topk, 1)
    
    # unsqueeze adds a tensor dim of 1 for batch size at pos 0
    # must be a FloatTensor to work with my model

    return top_ps.detach().numpy().reshape(-1), top_classes.numpy().reshape(-1)


model = loadmodel(dir_path+'/'+args.checkpoint)
idx_to_cat = {val: key for key, val in model.class_to_idx.items()}

probs, classes = predict(args.img_filepath, model, args.user_top_k, args.tryGPU)

if args.category_names is not None:
    import json
    with open(dir_path+'/'+args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes_labels = [cat_to_name[idx_to_cat[x]] for x in classes]
else:
    classes_labels = [idx_to_cat[x] for x in classes]
    
print('Predictions:')
for x in range(args.user_top_k):
    print(f"class: {classes_labels[x]}, {probs[x]*100:.2f}%")
    
    



