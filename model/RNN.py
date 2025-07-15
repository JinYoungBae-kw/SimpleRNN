import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size = 임베딩 차원의 수
        # hidden_size = hidden_state의 길이, 이 길이와 input_size를 기반으로 연산을 수행한다.
        # output_size = 출력 길이
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # input -> h 로 가는 층
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # h -> y로 가는 층
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.input_to_hidden.weight)
        nn.init.zeros_(self.input_to_hidden.bias)

        # tanh 함수
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 배치당 문장 수
        batch_size = x.size(0)
        # 0으로 초기화한 1*hidden_size 행렬 생성
        h = torch.zeros(batch_size, self.hidden_size)

        # x의 1번째 요소 크기만큼 반복문 진행
        # 즉, 시퀀스의 최대 길이수만큼 반복 -> 모든 시퀀스를 돌겠다는 뜻
        for t in range(x.size(1)):
            #  수식에서 x와 h가 묶여있는 부분을 표현하는 행렬
            combined = torch.cat((x[:, t, :], h), dim=1)
            # x -> h로 가는 층을 통과
            h = self.tanh(self.input_to_hidden(combined))
        # 위에서 구한 이전 시점의 h값을 h -> y로 가는 층을 통과시켜 y 계산
        y = self.hidden_to_output(h)
        return y