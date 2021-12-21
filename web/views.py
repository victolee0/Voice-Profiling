from django.http.response import HttpResponse
from django.shortcuts import render
from django.http import JsonResponse
import torch
import librosa
#from rnnoise_wrapper import RNNoise
import torchaudio
import io
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

class FeatureSharing1(nn.Module):
    def __init__(self, num_channels=3, channel_1=16, channel_2=64, num_classes=6):
        super(FeatureSharing1, self).__init__()
        self.num_channels = num_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.num_channels, self.channel_1, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(self.channel_1, self.channel_2, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(channel_1)
        self.bn2 = nn.BatchNorm2d(channel_2)

        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AvgPool2d((3, 99))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32, bias=True)

        self.fc2 = nn.Linear(32, 5, bias=True)
        self.fc3 = nn.Linear(32, 2, bias=True)
        self.fc4 = nn.Linear(32, 32, bias=True)
        self.fc5 = nn.Linear(32, 6, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout2d(F.relu(self.bn1(x)))
        x = self.pool(x)

        x = self.conv2(x)
        x = self.dropout2d(F.relu(self.bn2(x)))
        x = self.pool(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), 64)
        x = self.dropout(F.relu(self.fc1(x)))
        age = self.fc2(x)
        gender = self.fc3(x)
        dialect = self.fc5(self.fc4(x))

        return [age, gender, dialect]


# Create your views here.
def main(request):
    return render(request, 'main.html', {})

def predict(request):
    if request.method == 'POST':

        #데이터 입력값 받기
        #denoiser=RNNoise()

        blob = request.FILES

        print(blob)
        #y, sr = torchaudio.load(blob['audioFile'].file)
        b = blob['audioFile'].file.getvalue()
        #print(wave.open(blob['audioFile'].file))
        print(type(b))
        y, sr = torchaudio.load(io.BytesIO(b))
        print(sr)
        print(y.size())
        if y.size()[0] == 2:
            y = y[0]

        #tmp = {'Directory': directory} #data.append(tmp, ignore_index=True) #file_dir = data['Directory']
        #denoiser = RNNoise()
        #audio = denoiser.read_wav(io.BytesIO(b))######
        #sr = audio.frame_rate
        #denoised_audio = denoiser.filter(audio)
        #signal = torch.tensor(denoised_audio.get_array_of_samples()) / 2**15
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

        model=FeatureSharing1()
        model.load_state_dict(torch.load('model.pt', map_location=device))
        model.eval()

        with torch.no_grad():
            x = torch.tensor(mfcc_result).view(1, 3, 14, 400)
            x = x.to(device=device, dtype=dtype)
            model = model.to(device=device)
            age_score, gender_score, dialect_score = model(x)
            age = age_score.max(1)
            gender = gender_score.max(1)
            dialect = dialect_score.max(1)

        age_ = ["유아", "청소년", "청년", "중장년", "노년"]
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

        #데이터 입력값 받기
        #denoiser=RNNoise()
        print("OK")

        blob = request.POST.get('audioFile')

        results='test'

        return JsonResponse({'result':results,
                            'blob':blob[0]
                            })

'''
        #modcd ..el 가져오기
        ## 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
        model=FeatureSharing1()
        model=model.load_state_dict(torch.load('model.pt'))
        model.eval()

        with torch.no_grad():
            x = torch.tensor(data['input']).view(1, 3, 14, 400)
            x = x.to(device=device, dtype=dtype)
            model = model.to(device=device)
            age_score, gender_score, dialect_score = model(x)
            age = age_score.max(1)
            gender = gender_score.max(1)
            dialect = dialect_score.max(1)

        age_ = ["유아", "청소년", "청년", "중장년", "노년"]
        gender_ = ["여성", "남성"]
        dialect_ = ['수도권', '전라도', '경상도', '충청도', '강원도', '제주도']
        results=[]

        for i in range(len(age_)):
            if age[1].tolist()[0] == i:
                results.append(age_[i])

        for i in range(len(gender_)):
            if gender[1].tolist()[0] == i:
                results.append(gender_[i])

        for i in range(len(dialect_)):
            if dialect[1].tolist()[0] == i:
                results.append(dialect_[i])
'''
