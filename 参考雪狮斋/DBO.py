import copy
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import imageio
plt.rcParams['font.sans-serif']=['Microsoft YaHei']  # 修改字体为宋体
'''适应度函数Sphere'''


def F1(x):
    o = np.sum(np.square(x))
    return o


'''种群初始化'''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
    return X, lb, ub


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    pop = X.shape[0]
    X_new = np.zeros(X.shape)
    for i in range(pop):
        X_new[i, :] = X[index[i], :]
    return X_new


'''蜣螂滚球行为与舞蹈行为'''


def BRUpdate(X, XLast, pNum, GworstPosition):
    X_new = copy.copy(X)
    r2 = np.random.rand(1)
    dim = X.shape[1]
    b = 0.3
    for i in range(pNum):
        if r2 < 0.9:
            a = np.random.rand(1)
            if a > 0.1:
                a = 1
            else:
                a = -1
            X_new[i, :] = X[i, :] + b * np.abs(X[i, :] - GworstPosition[0,:]) + a * 0.1 * (
                XLast[i, :])  # Equation(1)
        else:
            aaa = np.random.randint(180, size=1)
            if aaa == 0 or aaa == 90 or aaa == 180:
                for j in range(dim):
                    X_new[i, j] = X[i, j]
            theta = aaa * math.pi / 180
            X_new[i, :] = X[i, :] + math.tan(theta) * np.abs(X[i, :] - XLast[i, :])  # Equation(2)
    # 防止产生蜣螂位置超出搜索空间
    for i in range(pNum):
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lb[j], ub[j])
    return X_new


'''蜣螂繁殖行为'''


def SPUpdate(X, pNum, t, Max_iter, fitness):
    X_new = copy.copy(X)
    dim = X.shape[1]
    R = 1 - t / Max_iter
    fMin = np.min(fitness)  # 找到X中最小的适应度
    bestIndex = np.argmin(fitness)  # 找到X中最小适应度的索引
    bestX = X[bestIndex, :]  # 找到X中具有最有适应度的蜣螂位置
    lbStar = bestX * (1 - R)
    ubStar = bestX * (1 + R)
    for j in range(dim):
        lbStar[j] = np.clip(lbStar[j], lb[j], ub[j])  # Equation(3)
        ubStar[j] = np.clip(ubStar[j], lb[j], ub[j])  # Equation(3)
    # XLb = swapfun(lbStar)
    # XUb = swapfun(ubStar)
    for i in range(pNum + 1, 12):  # Equation(4)
        X_new[i, :] = bestX + (np.random.rand(1, dim)) * (
                X[i, :] - lbStar + (np.random.rand(1, dim)) * (X[i, :] - ubStar))
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lbStar[j], ubStar[j])
    return X_new


'''蜣螂觅食行为'''


def FAUpdate(X, t, Max_iter, GbestPosition):
    X_new = copy.copy(X)
    dim = X.shape[1]
    R = 1 - t / Max_iter
    lbb = GbestPosition[0, :] * (1 - R)
    ubb = GbestPosition[0, :] * (1 + R)
    for j in range(dim):
        lbb[j] = np.clip(lbb[j], lb[j], ub[j])  # Equation(5)
        ubb[j] = np.clip(ubb[j], lb[j], ub[j])  # Equation(5)
    for i in range(13, 19):  # Equation(6)
        X_new[i, :] = X[i, :] + (np.random.rand(1, dim)) * (X[i, :] - lbb) + (np.random.rand(1, dim)) * (X[i, :] - ubb)
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lbb[j], ubb[j])
    return X_new


'''蜣螂偷窃行为'''


def THUpdate(X, t, Max_iter, GbestPosition, fitness):
    X_new = copy.copy(X)
    dim = X.shape[1]
    fMin = np.min(fitness)  # 找到X中最小的适应度
    bestIndex = np.argmin(fitness)  # 找到X中最小适应度的索引
    bestX = X[bestIndex, :]  # 找到X中具有最有适应度的蜣螂位置
    for i in range(20, pop):  # Equation(7)
        X_new[i, :] = GbestPosition[0, :] + np.random.randn(1, dim) * (
                np.abs(X[i, :] - GbestPosition[0, :]) + np.abs(X[i, :] - bestX)) / 2
        for j in range(dim):
            X_new[i, j] = np.clip(X_new[i, j], lb[j], ub[j])
    return X_new

'''蜣螂优化算法'''
def DBO(pop, dim, lb, ub, Max_iter, fun):
    """
        :param fun: 适应度函数
        :param pop: 种群数量
        :param Max_iter: 迭代次数
        :param lb: 迭代范围下界
        :param ub: 迭代范围上界
        :param dim: 优化参数的个数
        :return: GbestScore、GbestPosition、Curve : 适应度值最小的值、对应的位置、迭代过程中的历代最优位置
    """
    P_percent = 0.2
    pNum = round(pop * P_percent)
    X, lb, ub = initial(pop, dim, ub, lb)
    fitness = CaculateFitness(X, fun)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    XLast = X  # X(t-1)
    GbestScore = copy.copy(fitness[0])
    GbestPosition = np.zeros([1, dim])
    GbestPosition[0, :] = copy.copy(X[0, :])

    GworstScore = copy.copy(fitness[-1])
    GworstPosition = np.zeros([1, dim])
    GworstPosition[0, :] = copy.copy(X[-1, :])

    Curve = np.zeros([Max_iter, 1])
    Space = np.zeros([Max_iter, dim])
    for t in range(Max_iter):
        BestF = fitness[0]
        X = BRUpdate(X, XLast, pNum, GworstPosition)  # 滚球和舞蹈行为
        fitness = CaculateFitness(X, fun)  # 重新计算并排序
        X = SPUpdate(X, pNum, t, Max_iter, fitness)
        X = FAUpdate(X, t, Max_iter, GbestPosition)
        fitness = CaculateFitness(X, fun)  # 重新计算并排序
        X = THUpdate(X, t, Max_iter, GbestPosition, fitness)
        fitness = CaculateFitness(X, fun)  # 计算适应度值
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        XLast = X
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0, :] = copy.copy(X[0, :])
        Curve[t] = GbestScore
        Space[t] = GbestPosition[0, :]
        if (t) % 50 == 0:
            print("第%d代搜索的结果为:%f" % (t, GbestScore[0]))
    return GbestScore, GbestPosition, Curve, Space


if __name__ == '__main__':
    pop = 30
    dim = 2
    Max_iter = 500
    lb = -100 * np.ones([dim, 1])
    ub = 100 * np.ones([dim, 1])
    fun = F1
    print("DBO算法开始寻优")
    GbestScore, GbestPositon, Curve, Space = DBO(pop, dim, lb, ub, Max_iter, fun)
    print(GbestScore)
    print(GbestPositon)


    ax = plt.axes()
    # set limits
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.xlabel("x1")
    plt.ylabel("x2")
    image_list = []
    for i in range(Max_iter):
        # add something to axes
        if i % 5 == 0:
            ax.scatter(Space[i, 0], Space[i, 1])
            ax.plot(Space[i, 0], Space[i, 1], 'bo')
            plt.title("第%d次迭代" % (i + 1))
            # draw the plot
            plt.draw()
            plt.pause(0.001)
            plt.savefig('temp.png')
            image_list.append(imageio.imread('temp.png'))
    imageio.mimsave('pic.gif', image_list, duration=1)
    plt.show()
    # 适应度函数选择
