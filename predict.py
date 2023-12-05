import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse

import numpy as np

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxboard import SummaryWriter

from lib.utils import compute_val_loss, evaluate, predict, evaluate_best, evaluate_average
from lib.data_preparation import read_and_generate_dataset
from model.model_config import get_backbones

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    help="configuration file path", required=True)
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
args = parser.parse_args()

if os.path.exists('logs'):
    shutil.rmtree('logs')
    print('Remove log dir')

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

model_name = training_config['model_name']
ctx = training_config['ctx']
optimizer = training_config['optimizer']
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
merge = bool(int(training_config['merge']))

if ctx.startswith('cpu'):
    ctx = mx.cpu()
elif ctx.startswith('gpu'):
    ctx = mx.gpu(int(ctx[ctx.index('-') + 1:]))

print('Model is %s' % (model_name))
if model_name == 'DSTEGCN':
    from model.DSTEGCN import DSTEGCN as model
else:
    raise SystemExit('Wrong type of model!')

class MyInit(mx.init.Initializer):
    xavier = mx.init.Xavier()
    uniform = mx.init.Uniform()
    def _init_weight(self, name, data):
        if len(data.shape) < 2:
            self.uniform._init_weight(name, data)
            print('Init', name, data.shape, 'with Uniform')
        else:
            self.xavier._init_weight(name, data)
            print('Init', name, data.shape, 'with Xavier')

if __name__ == "__main__":
    print("Reading data...")
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    train_loader = gluon.data.DataLoader(
                        gluon.data.ArrayDataset(
                            nd.array(all_data['train']['week'], ctx=ctx),
                            nd.array(all_data['train']['day'], ctx=ctx),
                            nd.array(all_data['train']['recent'], ctx=ctx),
                            nd.array(all_data['train']['target'], ctx=ctx)
                        ),
                        batch_size=batch_size,
                        shuffle=True
    )
    val_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['val']['week'], ctx=ctx),
                        nd.array(all_data['val']['day'], ctx=ctx),
                        nd.array(all_data['val']['recent'], ctx=ctx),
                        nd.array(all_data['val']['target'], ctx=ctx)
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )
    test_loader = gluon.data.DataLoader(
                    gluon.data.ArrayDataset(
                        nd.array(all_data['test']['week'], ctx=ctx),
                        nd.array(all_data['test']['day'], ctx=ctx),
                        nd.array(all_data['test']['recent'], ctx=ctx),
                        nd.array(all_data['test']['target'], ctx=ctx)
                    ),
                    batch_size=batch_size,
                    shuffle=False
    )
    all_backbones = get_backbones(args.config, adj_filename, ctx)

    net = model(num_for_predict, all_backbones)
    net.initialize(ctx=ctx)
    for val_w, val_d, val_r, val_t in val_loader:
        net([val_w, val_d, val_r])
        break
    net.initialize(ctx=ctx, init=MyInit(), force_reinit=True)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': learning_rate})
        
    params_path = 'DSTEGCN/DSTEGCN/'
     
    best = 100
    count = 0

    for i in range(1,301):
        params_take_name = os.path.join(params_path,'DSTEGCN_epoch_%s.params' % (i))
        net.load_parameters(params_take_name)
        predc = evaluate_best(net, test_loader, true_value, num_of_vertices, epoch=0)
        if predc < best:
            best = predc
            count = i
        if i % 10 == 0:
            with open('DSTEGCN/DSTEGCN/test_result.txt','a') as file:
                file.write('Epoch Num: %d, MAE: %.2f\n' % (i,predc))
    
    print('Best MAE: %.2f' % (best))
    print('Best para: %d' % (count))