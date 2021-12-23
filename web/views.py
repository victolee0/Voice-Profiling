from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from librosa.feature.spectral import mfcc
from torch.autograd import Variable
import torch
import librosa
from rnnoise_wrapper import RNNoise
import torchaudio
import io
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32



class CLSTM_Fin(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=8,
                 hidden_size1=8, hidden_size2=16, hidden_size3=32,
                 num_classes1=4, num_classes2=2, num_classes3=6,
                 num_layers=2, batch_size=20):
        super(CLSTM_Fin, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_classes1 = num_classes1  # 연령
        self.num_classes2 = num_classes2  # 성별
        self.num_classes3 = num_classes3  # 지역

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(5, 5), padding=2, bias=True)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(3, 3), padding=1, bias=True)
        self.conv3 = nn.Conv2d(self.channel_2, 1, kernel_size=(3, 3), padding=1, bias=True)

        self.bn1 = nn.BatchNorm2d(self.channel_1)
        self.bn2 = nn.BatchNorm2d(self.channel_2)
        self.bn3 = nn.BatchNorm2d(1)

        self.LSTM1 = nn.LSTM(14, self.hidden_size1, self.num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(self.hidden_size1, self.hidden_size2, self.num_layers, batch_first=True)
        self.LSTM3 = nn.LSTM(self.hidden_size2, self.hidden_size3, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(400, num_classes1)
        self.fc2 = nn.Linear(400, num_classes2)
        self.fc3 = nn.Linear(400, num_classes3)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        ###### 음성 데이터 Feature 추출 ######

        self.h01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size1, device=x.device))
        self.c01 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size1, device=x.device))

        self.h02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size2, device=x.device))
        self.c02 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size2, device=x.device))

        self.h03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size3, device=x.device))
        self.c03 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size3, device=x.device))

        #self.h01 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size1))
        #self.c01 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size1))

        #self.h02 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size2))
        #self.c02 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size2))

        #self.h03 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size3))
        #self.c03 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size3))

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(x.size(0), -1, 400).transpose(2, 1)  # LSTM Input 완성

        ###### LSTM 학습 시작 ######

        # 1st Layer --> 연령
        out1, (h_1, c_1) = self.LSTM1(x, (self.h01, self.c01))  # 1st LSTM Hidden Layer
        h_t1 = torch.mean(out1.view(out1.size(0), out1.size(1), -1), dim=2)
        out_result1 = self.fc1(h_t1)

        # 2nd Layer --> 성별
        out2, (h_2, c_2) = self.LSTM2(out1, (self.h02, self.c02))  # 1st LSTM Hidden Layer
        h_t2 = torch.mean(out2.view(out2.size(0), out2.size(1), -1), dim=2)
        out_result2 = self.fc2(h_t2)

        # 3rd Layer --> 방언
        out3, (h_3, c_3) = self.LSTM3(out2, (self.h03, self.c03))  # 1st LSTM Hidden Layer
        h_t3 = torch.mean(out3.view(out3.size(0), out3.size(1), -1), dim=2)
        out_result3 = self.fc3(h_t3)

        return [out_result1, out_result2, out_result3]

class Shared_CNN(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=32, num_classes1=4, num_classes2=2, num_classes3=6):
        super(Shared_CNN, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))

        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.5)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9504, 1024, bias=True)
        self.fc2 = nn.Linear(9504, 1024, bias=True)
        self.fc3 = nn.Linear(9504, 1024, bias=True)

        self.fc10 = nn.Linear(1024, 128, bias=True)
        self.fc20 = nn.Linear(1024, 128, bias=True)
        self.fc30 = nn.Linear(1024, 128, bias=True)

        self.fc11 = nn.Linear(128, self.num_classes1, bias=True)
        self.fc21 = nn.Linear(128, self.num_classes2, bias=True)
        self.fc31 = nn.Linear(128, self.num_classes3, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc10.weight)
        nn.init.xavier_uniform_(self.fc20.weight)
        nn.init.xavier_uniform_(self.fc30.weight)
        nn.init.xavier_uniform_(self.fc11.weight)
        nn.init.xavier_uniform_(self.fc21.weight)
        nn.init.xavier_uniform_(self.fc31.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool(x)

        x = self.flatten(x)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = F.relu(self.fc3(x))

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        x1 = F.relu(self.fc10(x1))
        x2 = F.relu(self.fc20(x2))
        x3 = F.relu(self.fc30(x3))

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)

        score1 = self.fc11(x1)
        score2 = self.fc21(x2)
        score3 = self.fc31(x3)

        return [score1, score2, score3]


# Create your views here.
def main(request):
    return render(request, 'main.html', {})

def predict(request):
    if request.method == 'POST':

        denoiser=RNNoise()

        blob = request.FILES

        print(blob)
        b = blob['audioFile'].file.getvalue()
        model_type = request.POST['type']
        y, sr = torchaudio.load(io.BytesIO(b))

        if y.size()[0] == 2:
            y = y[0]

        vad=torchaudio.transforms.Vad(sr)
        signal=vad(y)
        n_fft = int(np.ceil(0.025 * sr))
        win_length = int(np.ceil(0.025 * sr))
        hop_length = int(np.ceil(0.01 * sr))
        audio_mfcc = torch.FloatTensor(librosa.feature.mfcc(y=signal.numpy().reshape(-1),
                                                                        sr=sr,
                                                                        n_mfcc=13,
                                                                        n_fft=n_fft,
                                                                        hop_length=hop_length))
        f0 = torch.FloatTensor(librosa.yin(y=signal.numpy().reshape(-1),
                                sr=sr,
                                frame_length = n_fft,
                                hop_length = hop_length,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C7')
                        ))
        f0[f0 > 500] = 0
        f0[torch.isnan(f0)] = 0
        delta1 = torch.FloatTensor(librosa.feature.delta(audio_mfcc, mode = 'constant'))
        delta2 = torch.FloatTensor(librosa.feature.delta(audio_mfcc, order=2, mode = 'constant'))
        f0_delta1 = torch.FloatTensor(librosa.feature.delta(f0, mode = 'constant')).view(1, -1)
        f0_delta2 = torch.FloatTensor(librosa.feature.delta(f0, order=2, mode = 'constant')).view(1, -1)
        audio_mfcc = np.concatenate((audio_mfcc, f0.reshape((1, -1))))
        delta1 = torch.cat((delta1, f0_delta1))
        delta2 = torch.cat((delta2, f0_delta2))
        audio_mfcc = torch.from_numpy(audio_mfcc)
        mfcc_result = torch.stack([audio_mfcc, delta1, delta2])
        if mfcc_result.shape[2] > 400:
            mfcc_result = mfcc_result[:, :, 0:400]
        else:
            mfcc_result = torch.cat([mfcc_result, torch.zeros(mfcc_result.shape[0], mfcc_result.shape[1], 400 - mfcc_result.shape[2])], dim=2)
        if model_type == 'HPS':
            print("HPS")
            model=Shared_CNN()
            model.load_state_dict(torch.load('HPS_CNN_SGD.pt', map_location=device))

        elif model_type == "CLSTM":
            print("CLSTM")

            model=CLSTM_Fin()
            model.load_state_dict(torch.load('CLSTM_ADAM_fin3.pt', map_location=device))

        model.eval()                
        with torch.no_grad():
            x = torch.tensor(mfcc_result).view(1, 3, 14, 400)
            x = x.to(device=device, dtype=dtype)
            model = model.to(device=device)
            age_score, gender_score, dialect_score = model(x)
            age = age_score.max(1)
            gender = gender_score.max(1)
            dialect = dialect_score.max(1)

        age_ = ["청소년", "청년", "중장년", "노년"]
        gender_ = ["여성", "남성"]
        dialect_ = ['수도권', '전라도', '경상도', '충청도', '강원도', '제주도']
        results={}

        for i in range(len(age_)):
            if age[1].tolist()[0] == i:
                results['age'] = age_[i]

        for i in range(len(gender_)):
            if gender[1].tolist()[0] == i:
                results['gender'] = gender_[i]

        for i in range(len(dialect_)):
            if dialect[1].tolist()[0] == i:
                results['dialect'] = dialect_[i]
        print(results)

    return HttpResponse(json.dumps(results))

def predict_app(request):
    if request.method == 'post':

       # denoiser=RNNoise()
        print("OK")

        blob = request.POST.get('audioFile')

        results='test'

        return JsonResponse({'result':results,
                            'blob':blob[0]
                            })

