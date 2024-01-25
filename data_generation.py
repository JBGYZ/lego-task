import torch
import numpy as np
import os
import math
import time
from torch.nn.functional import one_hot

def generate_data(tokenizer, n_var, voca_size = 8, batch_size=100):
    all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'][:voca_size]

    batch = []
    labels = []
    clause_order = []
    send_list = []
    reject_nb = 0
    total_nb = int (n_var * 2**n_var * math.factorial(voca_size)/math.factorial(voca_size-n_var))
    print('total_nb: ', total_nb)

    time_start = time.time()
    nb_counter = 0
    while nb_counter < batch_size:
        values = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses = []
        clauses.append('%s=+%d,' % (vars[0], values[0]))

        for i in range(1, n_var):
            modifier = '+' if values[i] == values[i-1] else '-'
            clauses.append('%s=%s%s,' % (vars[i], modifier, vars[i-1]))
            
        sent = ''
        label = []
        
        clause_idx = tuple(np.random.permutation(n_var))
        sent += ''.join([clauses[idx] for idx in clause_idx])
        label += [values[idx] for idx in clause_idx]
     
        order = torch.zeros(1, n_var, n_var)
        for i in range(n_var):
            order[0, i, clause_idx[i]] = 1
        
        if sent in send_list:
            reject_nb += 1
            if reject_nb % 100 == 0:
                print('reject_nb: ', reject_nb)
            continue
        else:
            send_list.append(sent)
            batch.append(tokenizer(sent))
            labels.append(values)
            clause_order.append(order)
            nb_counter += 1

    # print(len(labels))
    # print(labels[0].shape)
    labels = np.vstack(labels)
    # print(labels.shape)
    # print(np.vstack(labels).shape)
    print('reject_nb: ', reject_nb)
    time_end = time.time()
    print('time cost', time_end-time_start, 's')
    return torch.stack(batch), torch.LongTensor(labels), torch.cat(clause_order)

def make_lego_datasets(n_var, n_train, n_test, batch_size, voca_size, seed):
    # check whether the data file exists
    # if not, generate the data file
    if os.path.exists('lego_data_%d_%d_%d_%d_seed_%d.pt' % (n_var, n_train, n_test, voca_size, seed)):
        total_dataset = torch.load('lego_data_%d_%d_%d_%d_seed_%d.pt' % (n_var, n_train, n_test, voca_size, seed))
    else:
        tokenizer = CharTokenizer(voca_size = voca_size)
        n_total = n_train + n_test
        batch, labels, order = generate_data(tokenizer, n_var, voca_size, n_total)
        total_dataset = torch.utils.data.TensorDataset(batch, labels, order)
    trainset, testset = torch.utils.data.random_split(total_dataset, [n_train, n_test])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    # save the data file
    if not os.path.exists('lego_data_%d_%d_%d_%d_seed_%d.pt' % (n_var, n_train, n_test, voca_size, seed)):
        torch.save(total_dataset, 'lego_data_%d_%d_%d_%d_seed_%d.pt' % (n_var, n_train, n_test, voca_size, seed))
    return trainloader, testloader

class CharTokenizer:
    def __init__(self, voca_size = 8):
        all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'][:voca_size]
        all_vars += ['+', '-', '=', ',', '1', '0']
        all_vars = ''.join(all_vars)
        self.chars = sorted(list(set(all_vars)))
        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}

    def tokenize(self, text):
        return [self.char_to_int[char] for char in text]
    
    def decode(self, tokens_tensor):
        tokens = tokens_tensor.tolist()
        return ''.join([self.int_to_char[token] for token in tokens])

    def decode_one_hot(self, one_hot_tensor):
        token_tensors = torch.argmax(one_hot_tensor, dim=2)
        return [''.join([self.int_to_char[i.item()] for i in tokens]) for tokens in token_tensors]
    
    def one_hot_encode(self, text):
        tokens = self.tokenize(text)
        one_hot_ = one_hot(torch.tensor(tokens), num_classes=len(self.chars))
        return one_hot_
    
    def __call__(self, text):
        return self.one_hot_encode(text)

