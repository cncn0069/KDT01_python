import tensorflow as tf
from tensorflow.keras import layers, models

# 임의의 시계열 데이터 생성
import numpy as np
x_train = np.random.rand(1000, 10, 1)  # 1000개의 샘플, 10개의 타임스텝
y_train = np.random.rand(1000, 1)  # 1000개의 타겟 값

# 모델 구축
model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(10, 1)),
    layers.Dense(1)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

# 모델 훈련
model.fit(x_train, y_train, epochs=10)
