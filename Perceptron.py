import numpy as np
import time
import pickle


class Perceptron:
    # 생성자
    # thresholds : 임계값, 계산된  에측값을 비교하는
    # eta : 학습율
    # n_iner : 학습 횟수
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter

    # y = wx +b
    # 가중치 * 입력값의 총합 + b 구한다
    def net_input(self, X):
        a1 = np.dot(X, self.w_[1:]) + self.w_[0]
        return a1

    # 학습 함수
    def fit(self, X, y):
        # 가중치를 담는 변수 X 의 컬럼수 +1 해서 한자리는 b
        self.w_ = np.zeros(X.shape[1] + 1)
        # 예측 값 실제값 비교햇허 다른 값이 나온 수
        self.errors_ = []

        for _ in range(self.n_iter):
            # 실제 값과 예측값과 차이 난 개수
            errors = 0
            # 입력 받은 X,y 를 하나로 묶는다
            temp1 = zip(X, y)
            # 입력 값과 결과 값의 묶음을 가지고 반복.
            for xi, target in temp1:
                a1 = self.predict(xi)
                # 입력 값 ,예측값이 다르면 가중치를 조정한다
                if target != a1:
                    print("입력값 : ", xi)
                    print("target : ", target, "예측값 : ", a1)
                    print("업데이트 전 : ", self.w_)
                    update = self.eta * (target - a1)
                    print("업데이트 양 :", update)
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                    print("업데이트 후 : ", self.w_)
                    print("--------------------------------")

            # 실제 값과 예측 값이 다른 횟수 기록
            self.errors_.append(errors)

    # 예측 값 구하기
    def predict(self, X):
        a2 = np.where(self.net_input(X) > self.thresholds, 1, -1)
        return a2


def step1_learning(thresholds=0.0, eta=0.1, n_iter=10):
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([0, 0, 0, 1])
    ppn = Perceptron(thresholds=thresholds, eta=eta, n_iter=n_iter)
    s_time = time.time()
    ppn.fit(X, y)
    e_time = time.time()
    print("학습에 걸린 시간: ", e_time - s_time)
    print("학습 중에 오차난 수 : ", ppn.errors_)
    with open("./and_model.pickle", "wb") as f:
        pickle.dump(ppn, f)
    print("학습이 완료되었습니다.")


step1_learning()
