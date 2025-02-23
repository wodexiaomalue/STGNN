import argparse
import os
import torch
import random
from task import Task
import math
import time

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--adj_path', type=str, default='', help='data file')


parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


parser.add_argument('--seq_len', type=int, default=14, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=14, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=10, help='prediction sequence length')

parser.add_argument('--encoder_in', type=int, default=7, help='encoder input size')
parser.add_argument('--decoder_in', type=int, default=7, help='decoder input size')
parser.add_argument('--output', type=int, default=7, help='output size')

parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--n_fin', type=int, default=1)
parser.add_argument('--vex_dim', type=int, default=1)
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--moving_avg', type=int, default=11, help='window size of moving average')

parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
parser.add_argument('--predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

if __name__ == '__main__':

    args = parser.parse_args()

    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data_info = {'data': 'data_5m_in.csv', 'adj': 'adj_h1.csv', 'M2M': [80, 80, 80]}
    #data_info = {'data': 'pde/refined3.csv', 'adj': 'adj.csv', 'M2M': [13, 13, 13]}

    args.data_path = data_info['data']
    args.adj_path = data_info['adj']

    args.encoder_in, args.decoder_in, args.output = data_info['M2M']

    # args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    time = math.floor(time.time())
    ver_filename = 'time_{}_seq_l_{}_label_l_{}_pre_l_{}_d_model_{}_e_l_{}_d_l_{}'.format(time,
                                                                                          args.seq_len, args.label_len,
                                                                                          args.pred_len,
                                                                                          args.d_model, args.e_layers,
                                                                                           args.d_layers)
    ver_filename = 'time_{}'.format(time)
    task = Task(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(ver_filename))
    task.train(ver_filename)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(ver_filename))
    task.test(ver_filename)

    if args.predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(ver_filename))
        task.predict(ver_filename)

    torch.cuda.empty_cache()
