import numpy
import threading
import csv
from rpforest import RPForest


def leaves(predict, leaf):
    treeleaves = predict[0:leaf]
    return treeleaves


def missrateav(n, k, model, data, labels, leaf):
    miss = 0
    for i in range(n):
        count = 0
        predict = model.get_candidates(data[i], number=k, normalise=False)
        treeleaves = leaves(predict, leaf)
        for j in range(leaf):
            if labels[treeleaves[j]] == labels[i]:
                count = count + 1
        if count < k:
            miss = miss + k - count
    return miss / (n*k)


def discrepancyav(n, k, model, data):
    dist = 0
    for i in range(n):
        predict = model.get_candidates(data[i], number=k, normalise=False)
        dist = dist + numpy.linalg.norm(data[i] - data[predict[k-1]])
    return dist / n


def evaluate(data, labels, n, k, t, leaf, steps, runs, avmra, avdia):
    global rc
    for r in range(runs):
        mra = numpy.empty(steps, dtype=float)
        dia = numpy.empty(steps, dtype=float)
        for i in range(steps):
            model = RPForest(leaf_size=leaf, no_trees=t)
            model.fit(data, normalise=False)
            miss = 0
            mra[i] = missrateav(n, k, model, data, labels, leaf)
            dia[i] = discrepancyav(n, k, model, data)
            model.clear()
            if t == 100:
                t = 10
            elif i == 0:
                t = t + 10
            else:
                t = t + 20
        for w in range(steps):
            avmra[w] = avmra[w] + mra[w]
            avdia[w] = avdia[w] + dia[w]
        rc = rc + 1
        if rc == 1:
            print(str(rc) + ' Execution Out of ' + str(run) + ' Completed Successfully!')
        else:
            print(str(rc) + ' Executions Out of ' + str(run) + ' Completed Successfully!')


def normalise(avdia, avdiatemp, steps):
    minim = numpy.min(avdiatemp)
    maxim = numpy.max(avdiatemp)
    for s in range(steps):
        avdia[s] = (avdiatemp[s] - minim) / (maxim - minim)


print('The Supervised Version of KNN Search Using Random Projection Forests\nWritten in Python By Muhammad Rajabinasab')
print('#'.center(80, '#'))
data = numpy.loadtxt('1/full.csv', dtype=float, delimiter=',')
labels = list(csv.reader(open('1/fulllabels.csv', 'r')))
n = data.shape[0]
k = 5
t = 10
leaf = 20
steps = 6
run = 100
threads = 4
rc = 0
t1avmra = numpy.zeros(steps, dtype=float)
t1avdia = numpy.zeros(steps, dtype=float)
t2avmra = numpy.zeros(steps, dtype=float)
t2avdia = numpy.zeros(steps, dtype=float)
t3avmra = numpy.zeros(steps, dtype=float)
t3avdia = numpy.zeros(steps, dtype=float)
t4avmra = numpy.zeros(steps, dtype=float)
t4avdia = numpy.zeros(steps, dtype=float)
avmra = numpy.zeros(steps, dtype=float)
avdiatemp = numpy.zeros(steps, dtype=float)
avdia = numpy.zeros(steps, dtype=float)
print('Evaluating The Method...')
t1 = threading.Thread(target=evaluate, args=(data, labels, n, k, t, leaf, steps, int(run / threads), t1avmra, t1avdia))
t2 = threading.Thread(target=evaluate, args=(data, labels, n, k, t, leaf, steps, int(run / threads), t2avmra, t2avdia))
t3 = threading.Thread(target=evaluate, args=(data, labels, n, k, t, leaf, steps, int(run / threads), t3avmra, t3avdia))
t4 = threading.Thread(target=evaluate, args=(data, labels, n, k, t, leaf, steps, int(run / threads), t4avmra, t4avdia))
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()
for i in range(steps):
    avmra[i] = t1avmra[i] + t2avmra[i] + t3avmra[i] + t4avmra[i]
    avdiatemp[i] = t1avdia[i] + t2avdia[i] + t3avdia[i] + t4avdia[i]
for j in range(steps):
    avmra[j] = avmra[j] / rc
    avdiatemp[j] = avdiatemp[j] / rc
normalise(avdia, avdiatemp, steps)
print('Operation Completed Successfully!')
print('Average Missing Rate For Different Number of Trees in The Forest:')
print(avmra)
print('Average Discrepancy For Different Number of Trees in The Forest:')
print(avdia)
print('All Operations Completed Successfully')

