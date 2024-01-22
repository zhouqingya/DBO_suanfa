"""问题定义：
2维平面搜索问题，x1，x2属于[lb,ub],求x1与x2的平方和最小值
lb为搜索空间下界，ub上界。将每只蜣螂视为2维平面上的一个点，
该点的横纵坐标x1和x2即为该蜣螂的位置。
当前位置的好坏通过计算适应度fit=x平方+y平方的大小来评价
min fit = x平方 + y平方
假设种群共有50只蜣螂，d=2，n=50=pop
"""
#  载入所需要的包
import numpy as np
import random
import copy
import math


# 初始化蜣螂种群
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])  # 返回一个全是0的pop行，dim列的2维数组
    for i in range(pop):  # 使用循环填充数组X，range依次返回一个从0到pop-1的数
        X[i, :] = np.random.uniform(low=lb[0], high=ub[0], size=(1, dim))
        # X[i,:]表示读取X数组中第i行的所有列，生成在lb[0]和ub[0]之间的随机数，并将其作为第i行的值赋给X
    print(X)  # 要想输出X数组，需要把print放在return前面，不然没用
    return X, lb, ub  # 这里X就是一个pop行dim列的2维数组，填充的数是满足上下限要求随机数


'''
#  测试上面代码是否有问题
X1, _, _ = initial(50, 2, [10, 15], [0, 5])  # 获取第一个返回值，将lb和ub舍弃
# 所以现在的X就是我们随机生成那个数组
Y = np.square(X1)  # 直接对X进行平方操作，np.square就是对X中每个元素平方
print(Y)  # Y就是把随机生成的50行2列的数组中每个元素进行平均后的数组
'''
'''**************定义适应度函数************'''

'''定义适应度函数'''


def fun(X):
    O = 0
    for i in X:
        O += i ** 2
    return O


'''计算适应度函数'''


def CalculateFitness(X, fun):
    pop = X.shape[0]  # 返回X数组第一个维度的大小，即行数
    # 在 NumPy 中，shape 属性用于获取数组的维度信息，
    # 所以现在的pop就是种群的大小。pop=50
    fitness = np.zeros([pop, 1])  # 初始化适应度数组，生成一个pop行1列的全为0的数组
    for i in range(pop):
        fitness[i] = fun(X[i, :])  # 计算每个个体的适应度并保存在适应度数组中
    return fitness


'''将初始化得到的蜣螂种群按照适应度的大小进行排序，
则得到的具有最优适应度值的蜣螂为 X[0,:]'''

'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)  # axis=0 对适应度数组Fit沿着每列的方向进行排序
    index = np.argsort(Fit, axis=0)  # 获取排序后的索引
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)  # 创建一个与数组X同样形状大小的全为0的数组
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]  # 根据适应度排序后的索引重新排序位置
    return Xnew


'''以上是基础工作，接下来，
对初始生成的蜣螂种群按照滚球行为、跳舞行为、繁殖行为、觅食行为和偷窃行为的公式进行位置更新。'''

'''蜣螂滚球行为与舞蹈行为'''


def BRUpdate(X, XLast, pNum, GworstPosition):
    X_new = copy.copy(X)  # 复制输入的位置数组，以便对新数组进行修改而不影响原始数组
    r2 = np.random.rand(1)  # 生成一个范围在 [0, 1) 之间的随机数 r2
    dim = X.shape[1]  # 获取位置数组 X 的维度，即每个蜣螂的特征维度，
    # 获取这个形状元组的第二个元素，即n，表示X数组的列数
    b = 0.3
    for i in range(pNum):
        if r2 < 0.9:   # 若r2<0.9,无障碍物情况，进行——滚球操作
            a = np.random.rand(1)  # 又生成了一个[0,1)的随机数
            if a > 0.1:           #  判断这个随机数是否大于0.1
                a = 1            # 大于0.1，则赋值a=1，这里的a表示自然系数阿尔法
            else:
                a = -1
            X_new[i, :] = X[i, :] + b * np.abs(X[i, :] - GworstPosition[0, :]) + a * 0.1 * (
                XLast[i, :])  # Equation(1) 无障碍物时的更新公式
        #     后面写这个XLast=X，应该表达上一个位置，好像确实，最初的位置是X
        #  GworstPosition = np.zeros([1, dim]) 这是一个1行2列的全0数组
        #  这里有疑问
        #
        #

        else:   # 这里就是r2 >= 0.9时，即有障碍物情况，接下来就是——跳舞操作
            aaa = np.random.randint(180, size=1)
            if aaa == 0 or aaa == 90 or aaa == 180:
                for j in range(dim):
                    X_new[i, j] = X[i, j]
            theta = aaa * math.pi / 180
            X_new[i, :] = X[i, :] + math.tan(theta) * np.abs(X[i, :] - XLast[i, :])  # Equation(2)
            # 跳舞更新公式

    # 防止产生蜣螂位置超出搜索空间
    for i in range(pNum):   # 遍历每个蜣螂
        for j in range(dim):  # 遍历蜣螂的每个特征维度
            X_new[i, j] = np.clip(X_new[i, j], lb[j], ub[j])
    return X_new




