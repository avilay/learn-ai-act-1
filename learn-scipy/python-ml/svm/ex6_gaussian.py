import numpy as np
import matplotlib.pyplot as plt
from svm_analyzer import SvmAnalyzer


def load_data1():
    X = np.c_[(.4, -.7),
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              # --
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T

    y = [0] * 8 + [1] * 8

    return X, y


def load_data2():
    X = np.loadtxt('ex6data2_X.txt')
    y = np.loadtxt('ex6data2_y.txt')
    return X, y


def load_data3():
    X = np.array(
        [[1, 100], [1, 200], [1, 300], [1, 400], [2, 100], [2, 200], [3, 100], [3, 400], [4, 100], [4, 300], [4, 400],
         [4, 500], [5, 300], [5, 400], [5, 500]], dtype=np.float)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.float)
    return X, y


def load_data4():
    X = np.loadtxt('ex6data3_X.txt')
    y = np.loadtxt('ex6data3_y.txt')
    return X, y


def load_data4_cv():
    X_cv = np.loadtxt('ex6data3_cv_X.txt')
    y_cv = np.loadtxt('ex6data3_cv_y.txt')
    return X_cv, y_cv


def main():
    svma = SvmAnalyzer()

    svma.set_data(*load_data4())

    # plt.figure(1)
    # svma.vary_C(gamma=2)

    # plt.figure(2)
    # svma.vary_gamma(C=1)

    print("Starting with the error analysis")
    X_cv, y_cv = load_data4_cv()
    plots = {}
    for C in [1, 10, 100, 1000]:
        plots[C] = {'xplot': [], 'yplot': []}
        for gamma in range(2, 5):
            svma.learn(C, gamma)
            error = svma.calc_error(X_cv, y_cv)
            score = svma.get_clf().score(X_cv, y_cv)
            print('C = {}, gamma = {}: error = {}, score = {}'.format(C, gamma, error, score))
            plots[C]['xplot'].append(gamma)
            plots[C]['yplot'].append(error)

            # plt.figure(3)
            # plt.xlabel('gamma')
            # plt.ylabel('error')
            # for C in plots:
            # xplot = plots[C]['xplot']
            # 	yplot = plots[C]['yplot']
            # 	plt.plot(xplot, yplot, label='C = {}'.format(C))
            # plt.legend()

            # plt.figure(4)
            # svma.learn(C=10, gamma=16)
            # svma.plot()

            # plt.show()


if __name__ == '__main__':
    main()
