import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch # (batch num, batch size, n_step) (batch num, batch size)

def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict

class TextLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        # self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.input_size = emb_size
        self.hidden_size = n_hidden
        self.num_layers = 2
        self.bidirectional = False
        self.D = 2 if self.bidirectional else 1
        K = torch.sqrt(torch.tensor(1 / self.hidden_size))

        self.w_ih0 = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size, self.input_size)), requires_grad=True)
        self.w_hh0 = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size, self.hidden_size)), requires_grad=True)
        self.b_ih0 = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size)), requires_grad=True)
        self.b_hh0 = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size)), requires_grad=True)
        # ????????????????????????????????? ????????????
        for i in range(1, self.num_layers):
            exec("self.w_ih%s = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size, self.D*self.hidden_size)), requires_grad=True)"% i)
            exec("self.w_hh%s = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size, self.hidden_size)), requires_grad=True)" % i)
            exec("self.b_ih%s = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size)), requires_grad=True)" % i)
            exec("self.b_hh%s = nn.Parameter(torch.mul(K, torch.rand(4 * self.hidden_size)), requires_grad=True)" % i)

        if self.bidirectional:
            self.w_ih_r0 = self.w_ih0
            self.w_hh_r0 = self.w_hh0
            self.b_ih_r0 = self.b_ih0
            self.b_hh_r0 = self.b_hh0
            for i in range(1, self.num_layers):
                exec("self.w_ih_r%s = self.w_ih%s" %(i,i))
                exec("self.w_hh_r%s = self.w_hh%s" %(i,i))
                exec("self.b_ih_r%s = self.b_ih%s" %(i,i))
                exec("self.b_hh_r%s = self.b_hh%s" %(i,i))

        self.W = nn.Linear(n_hidden, n_class, bias=False) # ?????????????????? ?????????????????????
        self.b = nn.Parameter(torch.ones([n_class]))

    def LSTM_layer(self, input, initial_states, w_ih, w_hh, b_ih, b_hh):

        h0, c0 = initial_states
        batch_size, T, i_size = input.shape
        h_size = self.hidden_size
        pre_h = h0
        pre_c = c0

        batch_w_ih = w_ih.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_hh = w_hh.unsqueeze(0).tile(batch_size, 1, 1)

        output = torch.zeros(batch_size, T, self.hidden_size).to(device)

        for t in range(T):
            x = input[:, t, :]

            w_times_x = torch.bmm(batch_w_ih,x.unsqueeze(-1))
            w_times_x = w_times_x.squeeze(-1)
            w_times_pre_h = torch.bmm(batch_w_hh, pre_h.unsqueeze(-1))
            w_times_pre_h = w_times_pre_h.squeeze(-1)

            # ?????????
            i_t = torch.sigmoid(w_times_x[:, : h_size] + w_times_pre_h[:, : h_size] + b_ih[:h_size] + b_hh[:h_size])
            # ?????????
            f_t = torch.sigmoid(w_times_x[:,h_size : 2*h_size] + w_times_pre_h[:,h_size : 2*h_size] + b_ih[h_size : 2*h_size] + b_hh[h_size : 2*h_size])
            # ?????????
            g_t = torch.tanh(w_times_x[:,2*h_size : 3*h_size] + w_times_pre_h[:,2*h_size : 3*h_size] + b_ih[2*h_size : 3*h_size] + b_hh[2*h_size : 3*h_size])
            # ?????????
            o_t = torch.sigmoid(w_times_x[:,3*h_size :] + w_times_pre_h[:,3*h_size :] + b_ih[3*h_size :] + b_hh[3*h_size :])

            pre_c = f_t * pre_c +i_t * g_t
            pre_h = o_t * torch.tanh(pre_c)
            output[:, t, :] = pre_h # [batch_size, seq_len, hidden_size]

        return output, (pre_h, pre_c)

    def My_LSTM(self, input, input_size, hidden_size, num_layers = 1, bidirectional=False):
        D = self.D
        h_0 = torch.zeros(num_layers * D, len(input), hidden_size).to(device)
        c_0 = torch.zeros(num_layers * D, len(input), hidden_size).to(device)

        output0 = torch.zeros(input.shape[0], input.shape[1], D * hidden_size).to(device)
        output_list = []
                # [num_layers, batch_size, seq_len, D*hidden_size]
        h_n = torch.zeros(num_layers * D, input.shape[0], hidden_size).to(device)
        c_n = torch.zeros(num_layers * D, input.shape[0], hidden_size).to(device)

        for k in range(0, num_layers):
            if k==0:
                output0[:, :, :hidden_size], (h_n[k, :, :], c_n[k, :, :]) = \
                    self.LSTM_layer(input, (h_0[k, :, :],c_0[k, :, :]), self.w_ih0, self.w_hh0, self.b_ih0, self.b_hh0)
                if bidirectional:
                    output0[:, :, hidden_size:], (h_n[k+1, :, :], c_n[k+1, :, :]) = \
                        self.LSTM_layer(torch.flip(input, [1]), (h_0[k+1, :, :], c_0[k+1, :, :]), self.w_ih_r0, self.w_hh_r0, self.b_ih_r0, self.b_hh_r0)
                output_list.append(output0)
            else:
                exec("output%s = torch.zeros(input.shape[0], input.shape[1], D * hidden_size).to(device)"%k)
                exec("output%s[:, :, :hidden_size], (h_n[k*D, :, :], c_n[k*D, :, :]) = self.LSTM_layer(output0, (h_0[k*D, :, :], c_0[k*D, :, :]), "
                     "self.w_ih%s, self.w_hh%s, self.b_ih%s, self.b_hh%s)"%(k,k,k,k,k))
                if bidirectional:
                    exec("output%s[:, :, hidden_size:], (h_n[k*D+1, :, :], c_n[k*D+1, :, :]) = self.LSTM_layer(torch.flip(output0, [1]), (h_0[k*D+1, :, :], c_0[k*D+1, :, :]), "
                         "self.w_ih_r%s, self.w_hh_r%s, self.b_ih_r%s, self.b_hh_r%s)"%(k,k,k,k,k))
                exec("output_list.append(output%s)" %k)

        return output_list[-1], (h_n, c_n)

    def forward(self, X):
        X = self.C(X) # X : [batch_size, n_step, embeding size]
        # X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        # outputs, (_, _) = self.LSTM(X, (hidden_state, cell_state))

        outputs, (_, _) = self.My_LSTM(X, emb_size, n_hidden, num_layers=self.num_layers, bidirectional=self.bidirectional)
        outputs = outputs.transpose(0, 1) # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]

        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

def train():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)
            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 200 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target)*128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch+1}.ckpt')

def test(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model
    model.to(device)

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}????????????????????????")
    print('loss =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    batch_size = 128 # batch size
    learn_rate = 0.0005
    all_epoch = 5 #the all epoch for training
    emb_size = 256 #embeding size
    save_checkpoint_epoch = 5 # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt') # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    #print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  #n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]
    
    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)
    print("Using", device)

    print("\nTrain the LSTM????????????????????????")
    train()

    print("\nTest the LSTM????????????????????????")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test(select_model_path)
