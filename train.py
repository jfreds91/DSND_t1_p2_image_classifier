# The purpose of this program is to train a model on arbitrary image data

# expected arguments:
# directory - the top level directory containing a training and validation set

# optional arguments
# --save_dir save_directory - location to save model
# --arch "vgg13" - model architecture
# --learning_rate 0.01
# -- hidden_units 512
# -- epochs 20
# -- gpu

# imports
import argparse
import os
import sys
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

dir_path = os.path.dirname(os.path.realpath(__file__))

# Read in supported models
support_file = open('supported_models.txt', 'r')
supported_models = support_file.readlines()
supported_models = [s.strip() for s in supported_models]
support_file.close()

torch.cuda.empty_cache()

###############################################################################
################################### Argparse ##################################
###############################################################################

parser = argparse.ArgumentParser(description='NN training python executable')

# nonoptional arguments
parser.add_argument('load_directory', action='store', type=str,
                    help='The target image directory. Must contain test and\
                     validation subfolders. Type \'here\' to use this folder.')

# optional arguments
parser.add_argument('--save_dir', type=str, dest='save_dir', default=dir_path,
                    help='The directory to save model to. Default self')
parser.add_argument('--arch', type=str, dest='arch', default='densenet121',
                    choices = supported_models,
                    help='The type of model architecture to use')
parser.add_argument('--learning_rate', type=float, dest='learn_rate',
                    default = 0.003, help='Model learning rate. Default 0.003')
parser.add_argument('--hidden_units', type=int, dest='hidden_units',
                    default=500,
                    help='Number of hidden layers to build into the model. Default 500')
parser.add_argument('--epochs', type=int, dest = 'epochs', default = 1,
                    help='Number of epochs to use. Default 1')
parser.add_argument('--gpu', action='store_true', dest = 'tryGPU',
                    help='Attempt to run on GPU. Default False.')

args = parser.parse_args()

# all above variables now accessible through args.variablename
# error check, special cases
if args.load_directory == 'here':
    args.load_directory = dir_path.replace('\\','/')

if args.epochs > 10:
    print('Long runtime expected for ' + str(args.epochs) + ' epochs.')
    ans = input('Proceed? [y/n]: ')
    if ans == 'n':
        print('Exiting.')
        sys.exit(),
    else:
        print('Continuing...')
        
###############################################################################
################################# Model Training ##############################
###############################################################################

# directories
train_dir = args.load_directory + '/train'
valid_dir = args.load_directory + '/valid'
test_dir = args.load_directory + '/test'

# define transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

# Must make an intermediate dictionary because cat_to_idx gets sorted...
# TODO: delete?
#idx_to_cat = {val: key for key, val in train_data.class_to_idx.items()}
outputlen = len(train_data.class_to_idx)

# Build and train classifier
# Load pretrained NN
model_method = getattr(models, args.arch)
model = model_method(pretrained=True)

try:
    input_num = model.classifier[0].in_features
except:
    try:
        input_num = model.classifier[1].in_features
    except:
        try:
            input_num = model.classifier.in_features
        except:
            print('unrecognized model classification structure')
            sys.exit()

# Freeze parameters as not to backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_num, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(args.hidden_units, outputlen)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
model.classifier = classifier

# Check if GPU is available
if args.tryGPU:
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
        ans = input('GPU unavailable, continue with CPU? [y/n]')
        if ans == 'y':
            print('Continuing...')
        else:
            print('Exiting')
            sys.exit()
else:
    device = 'cpu'
    
criterion = nn.NLLLoss()

# only train the classifier parameters, feature params are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learn_rate)

print('running on: ' + str(device))

# send to device
model.to(device)

# ensure training mode
model.train()

steps = 0
running_loss = 0
print_every = 5
for e in range(args.epochs):
    for inputs, labels in trainloader:
        steps+=1
        
        # move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval() #turns dropout and optimization off for this step
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()
                    
                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {e+1}/{args.epochs} | "
                 f"Train loss: {running_loss/print_every:.3f} | "
                 f"Validation loss: {validation_loss/len(validationloader):.3f} | "
                 f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            
            running_loss = 0
            model.train()

# save checkpoint
model.class_to_idx = train_data.class_to_idx

state = {
        'state_dict': model.state_dict(),
        'label_idx': model.class_to_idx,
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'input_num': input_num
        }

torch.save(state, args.save_dir + '/checkpoint.pth')


















