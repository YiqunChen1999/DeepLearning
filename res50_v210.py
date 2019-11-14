"""
    @Author        Yiqun Chen
    @Time        2019-11-11
    @Info        res50 for cifar-10, using PyTorch
    @Modified
"""

"""
    SummaryWriter path
    5 .pt file name
    cuda name
"""

# import the main packages.
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Res50 import Res50, set_res50_params
#writer = SummaryWriter('runs/res50/v108')

# import auxiliary packages.
import os
import time
import pickle

# define the constants.
INFO = '****>>>>'
BATCH_SIZE = 256
DEVICE = torch.device('cuda:3')
FILENAME = 'res50_v109'
print(INFO, 'loading data from', FILENAME+'.pkl')
try:
    f = open(FILENAME+'.pkl', 'rb')
    data = pickle.load(f)
    EPOCH = data['epoch']
    LEARNING_RATE = data['lr']
    evaluating_accuracy_list = data['list']
    f.close()
    print(INFO, 'done!')
except:
    print(INFO, 'failed to load data')
    EPOCH = 0
    LEARNING_RATE = 0.1
    evaluating_accuracy_list = []
    print(INFO, 'set learning rate to', LEARNING_RATE)
writer = SummaryWriter('runs/res50/'+FILENAME)
try:
    f = open(FILENAME+'.log', 'a+')
    f.close()
except:
    f = open(FILENAME+'.log', 'wr')
    f.write(FILENAME)
    f.close()

def set_params(model, training_data_loader, testing_data_loader):
    print(INFO, 'setting params...')
    loss_fn = nn.CrossEntropyLoss()
    training_params = {
    'epochs': 64000, 
    'data': training_data_loader, 
    'device': DEVICE,
    'optimizer': torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE, 
        momentum=0.9, 
        weight_decay=0.0001),
    'loss_fn': loss_fn
    }
    evaluating_params = {
    'data': testing_data_loader,
    'device': DEVICE,
    'loss_fn': loss_fn,
    }
    print(INFO, 'done!')
    return training_params, evaluating_params

'''
    @Input        None
    @Output        training_data_loader: loading the training data; 
                testing_data_loader: loading the testing data;
                classes: the classes of the cifar-10
    @Usage        
        training_data_loader, testing_data_loader, classes = get_data()
'''
def get_data():
    print(INFO, 'loading the training and testing data....')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    

    # loading the training data.
    training_set = torchvision.datasets.CIFAR10(
        root='../data', 
        train=True, 
        download=False, 
        transform=transform_train)
    training_data_loader = torch.utils.data.DataLoader(
        training_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2)

    # loading the testing data.
    testing_set = torchvision.datasets.CIFAR10(
        root='../data', 
        train=False, 
        download=False, 
        transform=transform_test)
    testing_data_loader = torch.utils.data.DataLoader(
        testing_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
     'dog', 'frog', 'horse', 'ship', 'truck')

    print(INFO, 'done!')
    return training_data_loader, testing_data_loader, classes


def fit(model, training_params, evaluate_params, LEARNING_RATE):
    device = training_params['device']
    epochs = training_params['epochs']
    optimizer = training_params['optimizer']
    loss_fn = training_params['loss_fn']
    now_time = time.asctime(time.localtime(time.time()))
    print(INFO, 'training start at', now_time)
    total_start_time = time.time()
    evaluating_accuracy = 0
    training_accuracy = 0
    for epoch in range(epochs):
        epoch += EPOCH
        model.train()
        start_time = time.time()
        training_loss = 0
        training_data = training_params['data']

        training_samples_num = 0
        training_correct_num = 0
        training_accuracy = 0
        for i, data in enumerate(training_data, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, training_predicted = torch.max(outputs, 1)
            training_samples_num += labels.size(0)
            training_correct_num += (training_predicted == labels).sum().item()
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            training_accuracy = 1.0 * training_samples_num / training_correct_num

            if i % 20 == 19:
                end_time = time.time()
                print('\repoch: %d\ttraining loss: %.8f\ttraining accuracy: %.5f\tcost time: %.3f'
                    % (epoch +1, 
                        training_loss/(i+1), 
                        training_accuracy,
                        end_time-start_time),
                    end='')        

        end_time = time.time()
        print('\repoch: %d\ttraining loss: %.8f\ttraining accuracy: %.5f\tcost time: %.3f' 
            % (epoch+1, 
                training_loss/(i+1), 
                training_accuracy, 
                end_time-start_time), 
            end='')
        writer.add_scalar('training loss:',
            training_loss/(i+1),
            epoch)
        

        model.eval()
        device = evaluate_params['device']
        testing_data = evaluate_params['data']
        evaluating_samples_num = 0
        evaluating_correct_num = 0
        with torch.no_grad():
            for i, data in enumerate(testing_data, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, evaluating_predicted = torch.max(outputs, 1)
                evaluating_samples_num += labels.size(0)
                evaluating_correct_num += (evaluating_predicted == labels).sum().item()
            evaluating_accuracy = 100.0 * evaluating_correct_num / evaluating_samples_num
            evaluating_accuracy_list.append(evaluating_accuracy)
            f = open(FILENAME+'.pkl', 'wb')
            data = {'epoch': epoch, 'list': evaluating_accuracy_list, 'lr': LEARNING_RATE}
            pickle.dump(data, f)
            f.close()
            print('\taccuracy: %.3f' % (evaluating_accuracy), '%')
            writer.add_scalar('evaluating accuracy', 
                evaluating_accuracy, 
                epoch)

        torch.save(model.state_dict(), FILENAME+'.pt')
        f = open(FILENAME+'.log', 'a+')    
        f.write('epoch:'+str(epoch+1)+' training loss:'+str(training_loss/(i+1))+' accuracy:'+str(evaluating_accuracy)+' lr:'+str(LEARNING_RATE)+'\n')
        training_loss = 0   

    total_end_time = time.time()
    now_time = time.asctime(time.localtime(time.time()))
    print('training finished at', now_time)
    print('total time used: %.3f' % (total_end_time-total_start_time))
    torch.save(model.state_dict(), FILENAME+'.pt')
    print(INFO, 'model saved successfully!')


def main():

    training_data_loader, testing_data_loader, classes = get_data()

    res50_params = set_res50_params()

    res50 = Res50(res50_params)

    print(INFO, 'preparing to load model from', FILENAME+'.pt...')
    try:
        res50.load_state_dict(torch.load(FILENAME+'.pt'))
        print(INFO, 'load model successfully!')
    except:
        print(INFO, 'failed to load model, maybe there is no file named', FILENAME+'.pt')

    training_params, evaluating_params = set_params(
        res50,
        training_data_loader, 
        testing_data_loader)

    res50.to(DEVICE)
    fit(res50, training_params, evaluating_params, LEARNING_RATE)
    images, labels = next(iter(training_data_loader))
    writer.add_graph(res50, images.to(DEVICE))
    writer.close()
    print(INFO, 'log data saved!')


#if '__name__' == __main__:
main()
            











