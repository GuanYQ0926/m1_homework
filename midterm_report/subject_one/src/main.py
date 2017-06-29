import csv
import matplotlib.pyplot as plt
import pywt


def diff_points(ys):
    point_num = 20
    diff_list = []
    yss = []
    xss = []
    for i in range(len(ys) - 1):
        diff_list.append(abs(ys[i] - ys[i + 1]))
    index_list = range(len(ys) - 1)
    index_list = sorted(index_list, key=lambda i: diff_list[i])
    for i in range(point_num):
        temp_index = index_list[-(i + 1)]
        xss.append(temp_index)
        yss.append(ys[temp_index])
    return xss, yss


def main1():
    with open('../asset/line-3.csv', 'rb') as f:
        csv_contents = csv.reader(f)
        for row in csv_contents:
            contents = row
    xs = range(len(contents))
    ys = [float(i) for i in contents]  # raw data
    # yss
    step = 30
    yss = []  # average data
    for i in range(len(xs) - step):
        amount = 0
        for j in range(step):
            amount += ys[i + j]
        yss.append(amount / float(step))
    for i in range(step):
        yss.append(ys[-(step - i)])
    # feature point ysss xsss
    xsss, ysss = diff_points(ys)
    plt.plot(xs, ys, c='b', lw=0.5)
    plt.plot(xs, yss, c='r', lw=0.5)
    plt.scatter(xsss, ysss, marker='x', c='g')
    # plt.plot(xsss, ysss, 'go')
    plt.show()


def main2():
    with open('../asset/line-3.csv', 'rb') as f:
        csv_contents = csv.reader(f)
        for row in csv_contents:
            contents = row
    xs = range(len(contents))
    ys = [float(i) for i in contents]  # raw data
    xss, yss = diff_points(ys)
    # yss
    cA, cD = pywt.dwt(ys, 'db2')
    cx, cy = diff_points(cA)
    plt.figure(1)
    plt.plot(xs, ys, c='b', lw=0.5)
    plt.scatter(xss, yss, marker='x', c='g')
    plt.figure(2)
    plt.plot(range(len(cA)), cA, c='r', lw=0.5)
    plt.scatter(cx, cy, marker='x', c='g')
    plt.show()


if __name__ == '__main__':
    main2()
