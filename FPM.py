import pandas as pd
from itertools import combinations
import time
import sys
import os, psutil
import math
import csv

def Apriori():
    row_len = len(store_data)
    col_len = len(store_data.values[0])
    records = []  #for dataset
    for i in range(0, row_len):
        records.append([int(store_data.values[i, j]) for j in range(0, col_len) if int(store_data.values[i, j]) != 0])
    init = []
    for i in records:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print()
    print("Using APRIORI Algorithm")
    print()
    support_thold = T_value
    threshold = math.ceil(support_thold * len(records))
    print("Min Support:",threshold)

    from collections import Counter
    c = Counter()
    for i in init:
        for d in records:
            if (i in d):
                c[i] += 1
    for i in c:
        str([i]) + ": " + str(c[i])
    print()
    l = Counter()
    for i in c:
        if (c[i] >= threshold):
            l[frozenset([i])] += c[i]
    test=[]
    print("Frequent Pattern:")
    for i in l:
        print(str(list(i)) + ": " + str(l[i]))
        test.append(str(list(i)) + ": " + str(l[i]))
    test.sort()
    test.sort(key = len)

    for count in range(2, 10000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i + 1, len(temp)):
                t = temp[i].union(temp[j])
                if (len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in records:
                temp = set(q)
                if (i.issubset(temp)):
                    c[i] += 1

        l = Counter()
        for i in c:
            if (c[i] >= threshold):
                l[i] += c[i]
        for i in l:
            print(str(list(i)) + ": " + str(l[i]))
            test.append(str(list(i)) + ": " + str(l[i]))
    test.sort()
    test.sort(key = len)
    print()
    print("After lexicographical ordering : " + str(test))

def Apriori_n():
    row_len = len(store_data)
    col_len = len(store_data.values[0])
    records = []
    for i in range(0, row_len):
        records.append([int(store_data.values[i, j]) for j in range(0, col_len) if int(store_data.values[i, j]) != 0])
    print()
    init = []
    for i in records:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print("When Algoname =", AlgoName, ", DataSet =", DataSet, " , Threshold =", Thold)
    print()
    support_thold = T_value
    threshold = math.ceil(support_thold * len(records))
    print("Min Support:",threshold)

    from collections import Counter
    c = Counter()
    for i in init:
        for d in records:
            if (i in d):
                c[i] += 1
    for i in c:
     (str([i]) + ": " + str(c[i]))
    l = Counter()
    num=0
    for i in c:
        if (c[i] >= threshold):
            l[frozenset([i])] += c[i]
    for i in l:
        (str(list(i)) + ": " + str(l[i]))
        num=num+1
    print()
    pl = l
    pos = 1
    for count in range(2, 10000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i + 1, len(temp)):
                t = temp[i].union(temp[j])
                if (len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in records:
                temp = set(q)
                if (i.issubset(temp)):
                    c[i] += 1
        ("C" + str(count) + ":")
        for i in c:
            (str(list(i)) + ": " + str(c[i]))
        l = Counter()
        for i in c:
            if (c[i] >= threshold):
                l[i] += c[i]
        ("L" + str(count) + ":")
        for i in l:
            (str(list(i)) + ": " + str(l[i]))
            num = num +1
        if (len(l) == 0):
            break
        pl = l
        pos = count
    ("L" + str(pos) + ":")
    for i in pl:
        (str(list(i)) + ": " + str(pl[i]))
    print("Num of Frequent Patterns Generated :",num)
    print()

def Apriori_rt():
    row_len = len(store_data)
    col_len = len(store_data.values[0])
    print("Row and Column:", row_len, col_len)

    records = []  #for dataset
    for i in range(0, row_len):
        records.append([int(store_data.values[i, j]) for j in range(0, col_len) if int(store_data.values[i, j]) != 0])  # remove all 0 value which converted from nan

    print("Dataset:", records)  #print all data
    print()

    init = []  #for itemset
    for i in records:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print("Item set:", init)  #show 1 length itemset
    print()

    tot_time = []
    threshold_val = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(records))  #calculate min sup
    print("Min Support:", threshold)
    print()

    for i in range(0, 8):
        start = time.time()
        from collections import Counter
        c = Counter()
        for i in init:
            for d in records: #items count from dataset
                if (i in d):
                    c[i] += 1
        print("C1:")
        for i in c:
            print(str([i]) + ": " + str(c[i]))
        print()
        l = Counter()
        for i in c:
            if (c[i] >= threshold):
                l[frozenset([i])] += c[i]
        print("L1:")
        for i in l:
            print(str(list(i)) + ": " + str(l[i]))
        print()
        pl = l
        pos = 1
        for count in range(2, 10000):
            nc = set()
            temp = list(l)
            for i in range(0, len(temp)):
                for j in range(i + 1, len(temp)):
                    t = temp[i].union(temp[j])
                    if (len(t) == count):
                        nc.add(temp[i].union(temp[j]))
            nc = list(nc)
            c = Counter()
            for i in nc:
                c[i] = 0
                for q in records:
                    temp = set(q)
                    if (i.issubset(temp)):
                        c[i] += 1
            print("C" + str(count) + ":")
            for i in c:
                print(str(list(i)) + ": " + str(c[i]))
            print()
            l = Counter()
            for i in c:
                if (c[i] >= threshold):
                    l[i] += c[i]
            print("L" + str(count) + ":")
            for i in l:
                print(str(list(i)) + ": " + str(l[i]))
            print()
            if (len(l) == 0):
                break
            pl = l
            pos = count
        print("Result: ")
        print("L" + str(pos) + ":")
        for i in pl:
            print(str(list(i)) + ": " + str(pl[i]))
        print()

        end = time.time()
        tot_time.append(end - start)
        threshold_val.append(support_thold)
        support_thold = support_thold - 0.01

        print("Total time:", tot_time)
        print("Threshold value:", threshold_val)


    #Threshold in the X-axis vs run time in Y-axis
    from matplotlib import pyplot as plt
    plt.plot(threshold_val,tot_time, label='dataset1')
    plt.xlabel("Threshold")
    plt.ylabel("Time (s)")
    plt.title("Apriori")
    plt.show()

def Apriori_m():
    row_len = len(store_data)
    col_len = len(store_data.values[0])
    records = []
    for i in range(0, row_len):
        records.append([int(store_data.values[i, j]) for j in range(0, col_len) if int(store_data.values[i, j]) != 0])

    init = []
    for i in records:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)

    memory = []
    threshold_val = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(records))

    for i in range(0, 8):
        from collections import Counter
        c = Counter()
        for i in init:
            for d in records:
                if (i in d):
                    c[i] += 1

        for i in c:
            (str([i]) + ": " + str(c[i]))

        l = Counter()
        for i in c:
            if (c[i] >= threshold):
                l[frozenset([i])] += c[i]

        for i in l:
            (str(list(i)) + ": " + str(l[i]))

        for count in range(2, 10000):
            nc = set()
            temp = list(l)
            for i in range(0, len(temp)):
                for j in range(i + 1, len(temp)):
                    t = temp[i].union(temp[j])
                    if (len(t) == count):
                        nc.add(temp[i].union(temp[j]))
            nc = list(nc)
            c = Counter()
            for i in nc:
                c[i] = 0
                for q in records:
                    temp = set(q)
                    if (i.issubset(temp)):
                        c[i] += 1
            ("C" + str(count) + ":")
            for i in c:
                (str(list(i)) + ": " + str(c[i]))

            l = Counter()
            for i in c:
                if (c[i] >= threshold):
                    l[i] += c[i]
            ("L" + str(count) + ":")
            for i in l:
                (str(list(i)) + ": " + str(l[i]))

            if (len(l) == 0):
                break

        memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        threshold_val.append(support_thold)
        support_thold = support_thold - 0.01

        print("Threshold value:", threshold_val)
        print("Memory:",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2 ,"MB")


    #Threshold in the X-axis vs memory usage in Y-axis
    from matplotlib import pyplot as plt
    plt.plot(threshold_val,memory, label='dataset2')
    plt.xlabel("Threshold")
    plt.ylabel("Memory (mb)")
    plt.title("Apriori")
    plt.show()

def Apriori_o():
    row_len = len(store_data)
    col_len = len(store_data.values[0])

    records = []
    for i in range(0, row_len):
        records.append([int(store_data.values[i, j]) for j in range(0, col_len) if int(store_data.values[i, j]) != 0])

    init = []
    for i in records:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)

    memory = []
    tot_time = []
    threshold_val = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(records))

    start = time.time()

    from collections import Counter
    c = Counter()
    for i in init:
        for d in records:
            if (i in d):
                c[i] += 1

    for i in c:
        (str([i]) + ": " + str(c[i]))

    l = Counter()
    for i in c:
        if (c[i] >= threshold):
            l[frozenset([i])] += c[i]

    for i in l:
        (str(list(i)) + ": " + str(l[i]))

    pl = l
    pos = 1
    for count in range(2, 10000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i + 1, len(temp)):
                t = temp[i].union(temp[j])
                if (len(t) == count):
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)
        c = Counter()
        for i in nc:
            c[i] = 0
            for q in records:
                temp = set(q)
                if (i.issubset(temp)):
                    c[i] += 1
        ("C" + str(count) + ":")
        for i in c:
            (str(list(i)) + ": " + str(c[i]))

        l = Counter()
        for i in c:
            if (c[i] >= threshold):
                l[i] += c[i]
        ("L" + str(count) + ":")
        for i in l:
            (str(list(i)) + ": " + str(l[i]))

        if (len(l) == 0):
            break
        pl = l
        pos = count
    ("L" + str(pos) + ":")
    for i in pl:
        (str(list(i)) + ": " + str(pl[i]))

    end = time.time()
    tot_time.append(end - start)
    memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    threshold_val.append(support_thold)
    print("Threshold", "time", "Memory" )
    print(threshold_val,tot_time,psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

def FPGrowth():
    row_len = len(store_data)
    col_len = len(store_data.values[0])

    items = []
    for i in range(0, row_len):
        items.append([int(store_data.values[i, j]) for j in range(0, col_len) if
                      int(store_data.values[i, j]) != 0])

    init = []
    for i in items:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print()
    print("Using FP-growth Algorithm")
    print()

    support_thold = T_value
    threshold = math.ceil(support_thold * len(items))
    print("Min Support:", threshold)
    print()
    count = [dict() for items in range(len(store_data.values[0]) + 1)]  #creating list of dictionary
    s = []  #support count for each individual items
    for i in items:
        for j in i:
            s.append(j)
    for i in s:
        if i in count[1]:  #if item is present in dictionary
            count[1][i] = count[1][i] + 1
        else:  #if item is not present in dictionary
            count[1][i] = 1

    for i in count[1].copy():
        if (count[1][i] < threshold):
            count[1].pop(i)   #removing infrequent items

    a = list(count[1])
    item = [list() for i in range(len(items))]

    for i in range(0, len(items)):
        for j in range(len(items[i])):
            if (a.__contains__(items[i][j]) != 0):
                item[i].append(items[i][j])
    #print("Frequent Dataset: ", item)

    def sort(a):
        for i in range(len(a) - 1):
            for j in range(len(a) - i - 1):
                if (count[1][a[j]] < count[1][a[j + 1]]):
                    a[j], a[j + 1] = a[j + 1], a[j]

    #print("After sorting all transactions with Frequent Order:")
    for i in range(0, len(items)):
        if (len(item[i]) > 1):
            sort(item[i])
        #  print(item[i])
    class tree:
        def __init__(self, name, sup, parent):
            self.name = name
            self.sup = sup
            self.nodeLink = None
            self.parent = parent
            self.children = []

    def ispresent(node, name):
        f = -1
        for i in node.children:
            f = f + 1
            if (i.name == name):
                return f
        return -1

    lastocc = count[1].copy()
    for i in lastocc:
        lastocc[i] = None

    root = tree("root", -1, None)
    z = 0
    for i in item:
        current = root
        for j in range(len(i)):
            if (ispresent(current, i[j]) >= 0):
                current = current.children[ispresent(current, i[j])]
                current.sup = current.sup + 1
            else:
                child = tree(i[j], 1, current)
                current.children.append(child)
                t = current
                current = current.children[ispresent(current, i[j])]
                current.parent = t
                if (lastocc[current.name] == None):
                    lastocc[current.name] = current
                else:
                    current.nodeLink = lastocc[current.name]
                    lastocc[current.name] = current

        def value(a):
            a = str(a)
            a = a[:-2]
            a = a[2:]
            a = a[:-1]
            return a

    def singlepath(node, n):
        sup = node.sup
        path = []
        pathname = []
        current = node

        while (current.parent != None):  #path from current node to root
            path.append(current)
            pathname.append(current.name)
            current = current.parent
        #  print("Item Set:", (node.name))
        #  print("Path :", pathname)

        path.remove(node)
        pathname.remove(node.name)
        candidatepath = []
        temp_candidatepath = []
        #  print("Conditional Pattern Base:", pathname)

        #generate combinations
        a = (list(combinations(pathname, n)))
        for j in a:
            temp_candidatepath.append(tuple(sorted(j)))

        for j in temp_candidatepath:
            j = list(j)
            j.append(node.name)
            candidatepath.append(sorted(j))
        #  print("Pattern Generated :", candidatepath)

        for j in candidatepath:
            j = tuple(j)
            if j in count[n + 1]:
                count[n + 1][j] = count[n + 1][j] + sup
            else:
                count[n + 1][j] = sup
        #  print("Update counts of the generated itemsets:")
        #print(count)

        if (node.nodeLink != None):
            node = node.nodeLink
            singlepath(node, i)

    def frequent(n): #if itemset is frequent
        f = 0
        for i in count[n]:
            if (count[n][i] >= threshold):
                f = 1
        if (f == 1):
            return 1

        else:
            return 0
    test=[]
    for i in range(1, len(store_data.values[0]) + 1):
        if (frequent(i) == 1):
            for j in lastocc:
                singlepath(lastocc[j], i)

    print("Frequent Patterns Generated:")
    for z in range(len(store_data.values[0]) + 1):
        for i in count[z].copy():
            if (count[z][i] < threshold):
                count[z].pop(i)   #remove infrequent itemsets
        print(count[z])
        test.append(str(count[z]))
    print("Generated Patterns: " + str(test))

def FPGrowth_n():
    row_len = len(store_data)
    col_len = len(store_data.values[0])
    items = []
    for i in range(0, row_len):
        items.append([int(store_data.values[i, j]) for j in range(0, col_len) if
                        int(store_data.values[i, j]) != 0])
    print()
    #print("Dataset:", items)
    init = []
    for i in items:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print("When Algoname =", AlgoName, ", DataSet =", DataSet, " , Threshold =", Thold)
    print()
    support_thold = T_value
    threshold = math.ceil(support_thold * len(items))
    print("Min Support:", threshold)
    print()
    count = [dict() for items in range(len(store_data.values[0]) + 1)]
    s = []
    for i in items:
        for j in i:
            s.append(j)
    for i in s:
        if i in count[1]:
            count[1][i] = count[1][i] + 1
        else:
          count[1][i] = 1

    for i in count[1].copy():
        if (count[1][i] < threshold):
            count[1].pop(i)

    a = list(count[1])
    item = [list() for i in range(len(items))]

    for i in range(0, len(items)):
        for j in range(len(items[i])):
            if (a.__contains__(items[i][j]) != 0):
                item[i].append(items[i][j])
    def sort(a):
        for i in range(len(a) - 1):
            for j in range(len(a) - i - 1):
                if (count[1][a[j]] < count[1][a[j + 1]]):
                    a[j], a[j + 1] = a[j + 1], a[j]

    for i in range(0, len(items)):
        if (len(item[i]) > 1):
            sort(item[i])
           # print(item[i])

    class tree:
        def __init__(self, name, sup, parent):
            self.name = name
            self.sup = sup
            self.nodeLink = None
            self.parent = parent
            self.children = []

    def ispresent(node, name):
        f = -1
        for i in node.children:
            f = f + 1
            if (i.name == name):
                return f
        return -1

    lastocc = count[1].copy()
    for i in lastocc:
        lastocc[i] = None

    root = tree("root", -1, None)
    z = 0
    for i in item:
        current = root
        for j in range(len(i)):
            if (ispresent(current, i[j]) >= 0):
                current = current.children[ispresent(current, i[j])]
                current.sup = current.sup + 1
            else:
                child = tree(i[j], 1, current)
                current.children.append(child)
                t = current
                current = current.children[ispresent(current, i[j])]
                current.parent = t
                if (lastocc[current.name] == None):
                    lastocc[current.name] = current
                else:
                    current.nodeLink = lastocc[current.name]
                    lastocc[current.name] = current
        def value(a):
            a = str(a)
            a = a[:-2]
            a = a[2:]
            a = a[:-1]
            return a

    def singlepath(node, n):
        sup = node.sup
        path = []
        pathname = []
        current = node

        while (current.parent != None):
            path.append(current)
            pathname.append(current.name)
            current = current.parent
        path.remove(node)
        pathname.remove(node.name)
        candidatepath = []
        temp_candidatepath = []
        a = (list(combinations(pathname, n)))
        for j in a:
            temp_candidatepath.append(tuple(sorted(j)))

        for j in temp_candidatepath:
            j = list(j)
            j.append(node.name)
            candidatepath.append(sorted(j))
        for j in candidatepath:
            j = tuple(j)
            if j in count[n + 1]:
                count[n + 1][j] = count[n + 1][j] + sup
            else:
                count[n + 1][j] = sup

        if (node.nodeLink != None):
            node = node.nodeLink
            singlepath(node, i)
    def frequent(n):
        f = 0
        for i in count[n]:
            if (count[n][i] >= threshold):
                f = 1
        if (f == 1):
            return 1

        else:
            return 0
    for i in range(1, len(store_data.values[0]) + 1):
        if (frequent(i) == 1):
            for j in lastocc:
                singlepath(lastocc[j], i)
    n = 0
   # print("Frequent Patterns Generated:")
    for z in range(len(store_data.values[0]) + 1):
        for i in count[z].copy():
            if (count[z][i] < threshold):
                count[z].pop(i)
       # print(count[z])

        if ((len(count[z])) > 0):
            n = n + (len(count[z]))
    print("Num of Frequent Patterns Generated :", n)

def FPGrowth_rt():
    row_len = len(store_data)
    col_len = len(store_data.values[0])

    items = []  # for dataset
    for i in range(0, row_len):
        items.append([int(store_data.values[i, j]) for j in range(0, col_len) if
                        int(store_data.values[i, j]) != 0])

    init = []
    for i in items:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)

    tot_time = []
    threshold_val = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(items))
    print("Min Support:", threshold)
    print()

    for i in range(0,8):
        start = time.time()
        count = [dict() for items in range(len(store_data.values[0]) + 1)]
        s = []
        for i in items:
            for j in i:
                s.append(j)
        for i in s:
            if i in count[1]:
                count[1][i] = count[1][i] + 1
            else:
                count[1][i] = 1

        for i in count[1].copy():
            if (count[1][i] < threshold):
                count[1].pop(i)

        a = list(count[1])
        item = [list() for i in range(len(items))]
        for i in range(0, len(items)):
            for j in range(len(items[i])):
                if (a.__contains__(items[i][j]) != 0):
                    item[i].append(items[i][j])


        def sort(a):
            for i in range(len(a) - 1):
                for j in range(len(a) - i - 1):
                    if (count[1][a[j]] < count[1][a[j + 1]]):
                        a[j], a[j + 1] = a[j + 1], a[j]

        for i in range(0, len(items)):
            if (len(item[i]) > 1):
                sort(item[i])


        class tree:
            def __init__(self, name, sup, parent):
                self.name = name
                self.sup = sup
                self.nodeLink = None
                self.parent = parent
                self.children = []

        def ispresent(node, name):
            f = -1
            for i in node.children:
                f = f + 1
                if (i.name == name):
                    return f
            return -1

        lastocc = count[1].copy()
        for i in lastocc:
            lastocc[i] = None

        root = tree("root", -1, None)
        z = 0
        for i in item:
            current = root
            for j in range(len(i)):
                if (ispresent(current, i[j]) >= 0):
                    current = current.children[ispresent(current, i[j])]
                    current.sup = current.sup + 1
                else:
                    child = tree(i[j], 1, current)
                    current.children.append(child)
                    t = current
                    current = current.children[ispresent(current, i[j])]
                    current.parent = t
                    if (lastocc[current.name] == None):
                        lastocc[current.name] = current
                    else:
                        current.nodeLink = lastocc[current.name]
                        lastocc[current.name] = current

        def value(a):
            a = str(a)
            a = a[:-2]
            a = a[2:]
            a = a[:-1]
            return a
        def singlepath(node, n):
            c = 0
            sup = node.sup
            path = []
            pathname = []
            current = node

            while (current.parent != None):
                path.append(current)
                pathname.append(current.name)
                current = current.parent

            path.remove(node)
            pathname.remove(node.name)
            candidatepath = []
            temp_candidatepath = []

            a = (list(combinations(pathname, n)))
            for j in a:
                temp_candidatepath.append(tuple(sorted(j)))

            for j in temp_candidatepath:
                j = list(j)
                j.append(node.name)
                candidatepath.append(sorted(j))

            for j in candidatepath:
                j = tuple(j)
                if j in count[n + 1]:
                    count[n + 1][j] = count[n + 1][j] + sup
                else:
                    count[n + 1][j] = sup

            if (node.nodeLink != None):
                node = node.nodeLink
                singlepath(node, i)

        def frequent(n):
            f = 0
            for i in count[n]:
                if (count[n][i] >= threshold):
                    f = 1
            if (f == 1):
                return 1
            else:
                return 0

        for i in range(1, len(store_data.values[0]) + 1):
            if (frequent(i) == 1):
                for j in lastocc:
                    singlepath(lastocc[j], i)

        for z in range(len(store_data.values[0]) + 1):
            for i in count[z].copy():
                if (count[z][i] < threshold):
                    count[z].pop(i)

        end = time.time()
        tot_time.append(end - start)
        threshold_val.append(support_thold)
        support_thold = support_thold - 0.01

        print("Total time:", tot_time)
        print()
        print("Threshold value:", threshold_val)

    from matplotlib import pyplot as plt
    plt.plot(threshold_val, tot_time, label=' dataset')
    plt.xlabel("Threshold")
    plt.ylabel("Time (s)")
    plt.show()

def FPGrowth_m():
    row_len = len(store_data)
    col_len = len(store_data.values[0])

    items = []
    for i in range(0, row_len):
        items.append([int(store_data.values[i, j]) for j in range(0, col_len) if
                        int(store_data.values[i, j]) != 0])
    init = []
    for i in items:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    print("Item set:", init)
    print()

    threshold_val = []
    memory = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(items))
    print("Min Support:", threshold)
    print()


    for i in range(0,8):
        count = [dict() for items in range(len(store_data.values[0]) + 1)]
        s = []
        for i in items:
            for j in i:
                s.append(j)
        for i in s:
            if i in count[1]:
                count[1][i] = count[1][i] + 1
            else:
                count[1][i] = 1
       # print("C1:", count)  # c1
        for i in count[1].copy():
            if (count[1][i] < threshold):
                count[1].pop(i)

        a = list(count[1])
        item = [list() for i in range(len(items))]
        c = 0
        for i in range(0, len(items)):
            for j in range(len(items[i])):
                if (a.__contains__(items[i][j]) != 0):
                    item[i].append(items[i][j])

        def sort(a):
            for i in range(len(a) - 1):
                for j in range(len(a) - i - 1):
                    if (count[1][a[j]] < count[1][a[j + 1]]):
                        a[j], a[j + 1] = a[j + 1], a[j]
       # print("After sorting all transactions with Frequent Order:")
        for i in range(0, len(items)):
            if (len(item[i]) > 1):
                sort(item[i])
               # print(item[i])
        class tree:
            def __init__(self, name, sup, parent):
                self.name = name
                self.sup = sup
                self.nodeLink = None
                self.parent = parent
                self.children = []

        def ispresent(node, name):
            f = -1
            for i in node.children:
                f = f + 1
                if (i.name == name):
                    return f
            return -1

        lastocc = count[1].copy()
        for i in lastocc:
            lastocc[i] = None

        root = tree("root", -1, None)
        z = 0
        for i in item:
            current = root
            for j in range(len(i)):
                if (ispresent(current, i[j]) >= 0):
                    current = current.children[ispresent(current, i[j])]
                    current.sup = current.sup + 1
                else:
                    child = tree(i[j], 1, current)
                    current.children.append(child)
                    t = current
                    current = current.children[ispresent(current, i[j])]
                    current.parent = t
                    if (lastocc[current.name] == None):
                        lastocc[current.name] = current
                    else:
                        current.nodeLink = lastocc[current.name]
                        lastocc[current.name] = current

        def value(a):
            a = str(a)
            a = a[:-2]
            a = a[2:]
            a = a[:-1]
            return a

        print()
        def singlepath(node, n):
            sup = node.sup
            path = []
            pathname = []
            current = node

            while (current.parent != None):
                path.append(current)
                pathname.append(current.name)
                current = current.parent
           # print("Item Set:", (node.name))
           # print("Path :", pathname)

            path.remove(node)
            pathname.remove(node.name)
            candidatepath = []
            temp_candidatepath = []
          #  print("Conditional Pattern Base:", pathname)

            a = (list(combinations(pathname, n)))
            for j in a:
                temp_candidatepath.append(tuple(sorted(j)))
            # print("temp_candidatepath",temp_candidatepath)

            for j in temp_candidatepath:
                j = list(j)
                j.append(node.name)
                candidatepath.append(sorted(j))
          #  print("Pattern Generated :", candidatepath)

            for j in candidatepath:
                j = tuple(j)
                if j in count[n + 1]:
                    count[n + 1][j] = count[n + 1][j] + sup
                else:
                    count[n + 1][j] = sup
            #print("Update counts of the generated itemsets :")
            #print(count)

            if (node.nodeLink != None):
                node = node.nodeLink
                singlepath(node, i)

        def frequent(n):
            f = 0
            for i in count[n]:
                if (count[n][i] >= threshold):
                    f = 1
            if (f == 1):
                return 1
            else:
                return 0

        for i in range(1, len(store_data.values[0]) + 1):
            if (frequent(i) == 1):
                for j in lastocc:
                    singlepath(lastocc[j], i)
       # print("Frequent Patterns Generated:")
        for z in range(len(store_data.values[0]) + 1):
            for i in count[z].copy():
                if (count[z][i] < threshold):
                    count[z].pop(i)
                 #   print(count[z])

        memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        threshold_val.append(support_thold)
        support_thold = support_thold - 0.01

        print("Threshold value:", threshold_val)
        print("Memory:",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2,"MB")

    #Threshold in the X-axis vs Memory usage in Y-axis
    from matplotlib import pyplot as plt
    plt.plot(threshold_val,memory, label='dataset4')
    plt.xlabel("Threshold")
    plt.ylabel("Memory (mb)")
    plt.title("FP Growth")
    plt.show()

def FPGrowth_o():
    row_len = len(store_data)
    col_len = len(store_data.values[0])

    items = []
    for i in range(0, row_len):
        items.append([int(store_data.values[i, j]) for j in range(0, col_len) if
                      int(store_data.values[i, j]) != 0])

    init = []
    for i in items:
        for q in i:
            if (q not in init):
                init.append(q)
    init = sorted(init)
    memory = []
    tot_time = []
    threshold_val = []
    support_thold = T_value
    threshold = math.ceil(support_thold * len(items))

    start = time.time()
    count = [dict() for items in range(len(store_data.values[0]) + 1)]
    s = []
    for i in items:
        for j in i:
            s.append(j)
    for i in s:
        if i in count[1]:
            count[1][i] = count[1][i] + 1
        else:
            count[1][i] = 1

    for i in count[1].copy():
        if (count[1][i] < threshold):
            count[1].pop(i)

    a = list(count[1])
    item = [list() for i in range(len(items))]

    for i in range(0, len(items)):
        for j in range(len(items[i])):
            if (a.__contains__(items[i][j]) != 0):
                item[i].append(items[i][j])

    def sort(a):
        for i in range(len(a) - 1):
            for j in range(len(a) - i - 1):
                if (count[1][a[j]] < count[1][a[j + 1]]):
                    a[j], a[j + 1] = a[j + 1], a[j]

    for i in range(0, len(items)):
        if (len(item[i]) > 1):
            sort(item[i])

    class tree:
        def __init__(self, name, sup, parent):
            self.name = name
            self.sup = sup
            self.nodeLink = None
            self.parent = parent
            self.children = []

    def ispresent(node, name):
        f = -1
        for i in node.children:
            f = f + 1
            if (i.name == name):
                return f
        return -1

    lastocc = count[1].copy()
    for i in lastocc:
        lastocc[i] = None

    root = tree("root", -1, None)
    z = 0
    for i in item:
        current = root
        for j in range(len(i)):
            if (ispresent(current, i[j]) >= 0):
                current = current.children[ispresent(current, i[j])]
                current.sup = current.sup + 1
            else:
                child = tree(i[j], 1, current)
                current.children.append(child)
                t = current
                current = current.children[ispresent(current, i[j])]
                current.parent = t
                if (lastocc[current.name] == None):
                    lastocc[current.name] = current
                else:
                    current.nodeLink = lastocc[current.name]
                    lastocc[current.name] = current

        def value(a):
            a = str(a)
            a = a[:-2]
            a = a[2:]
            a = a[:-1]
            return a

    def singlepath(node, n):
        sup = node.sup
        path = []
        pathname = []
        current = node

        while (current.parent != None):
            path.append(current)
            pathname.append(current.name)
            current = current.parent
        #  print("Item Set:", (node.name))
        #  print("Path :", pathname)

        path.remove(node)
        pathname.remove(node.name)
        candidatepath = []
        temp_candidatepath = []
        #  print("Conditional Pattern Base:", pathname)

        a = (list(combinations(pathname, n)))
        for j in a:
            temp_candidatepath.append(tuple(sorted(j)))

        for j in temp_candidatepath:
            j = list(j)
            j.append(node.name)
            candidatepath.append(sorted(j))
        #  print("Pattern Generated :", candidatepath)
        for j in candidatepath:
            j = tuple(j)
            if j in count[n + 1]:
                count[n + 1][j] = count[n + 1][j] + sup
            else:
                count[n + 1][j] = sup
        #  print("Update counts of the generated itemsets :")
        #print(count)
        if (node.nodeLink != None):
            node = node.nodeLink
            singlepath(node, i)

    def frequent(n):
        f = 0
        for i in count[n]:
            if (count[n][i] >= threshold):
                f = 1
        if (f == 1):
            return 1
        else:
            return 0
    test=[]
    for i in range(1, len(store_data.values[0]) + 1):
        if (frequent(i) == 1):
            for j in lastocc:
                singlepath(lastocc[j], i)

    for z in range(len(store_data.values[0]) + 1):
        for i in count[z].copy():
            if (count[z][i] < threshold):
                count[z].pop(i)
            (count[z])
        test.append(str(count[z]))


    end = time.time()
    tot_time.append(end - start)
    memory.append(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    threshold_val.append(support_thold)
    print("Threshold", "time", "Memory")
    print(threshold_val, tot_time, psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


if __name__ == '__main__':
    n = len(sys.argv)
    AlgoName = "AP"
    DataSet = "Toy.txt"
    Thold = 0.5

    if n > 1:
        for i in range(1, n):
            param = sys.argv[i]

            if param == "-a":
                AlgoName = sys.argv[i+1]
                if (AlgoName != "AP") and (AlgoName != "FP"):
                    print("Wrong input")
                    AlgoName = "AP"

            if param == "-d":
                DataSet = sys.argv[i+1]

                if DataSet == "mushroom.txt":
                    store_data = pd.read_csv('C:\\Users\\Shabrina Shara\\Downloads\\data\\mushroom.txt', sep=" ",header=None)
                    store_data.fillna(0, inplace=True)
                    store_data.head()
                    store_data = store_data.astype(int)
                elif DataSet == "chess.txt":
                    store_data = pd.read_csv('C:\\Users\\Shabrina Shara\\Downloads\\data\\chess.txt', sep=" ",header=None)
                    store_data.fillna(0, inplace=True)
                    store_data.head()
                    store_data = store_data.astype(int)
                elif DataSet == "kosarak.txt":
                    store_data = pd.read_csv('C:\\Users\\Shabrina Shara\\Downloads\\data\\kosarak.txt', header=None)
                    store_data.fillna(0, inplace=True)
                    store_data.head()
                    store_data = store_data.astype(int)
                elif DataSet == "retail.txt":
                    store_data = pd.read_csv('C:\\Users\\Shabrina Shara\\Downloads\\data\\retail.txt', header=None)
                    store_data.fillna(0, inplace=True)
                    store_data.head()
                    store_data = store_data.astype(int)
                else:
                    DataSet == "Toy.txt"
                    store_data = pd.read_csv('C:\\Users\\Shabrina Shara\\Downloads\\data\\Toy.txt', header=None)
                    store_data.fillna(0, inplace=True)
                    store_data.head()
                    store_data = store_data.astype(int)


            if param == "-t":
                Thold = float(sys.argv[i+1])
                T_value=Thold


            if (param == "-m") and AlgoName == "AP":
                Apriori_m()

            if (param == "-m") and AlgoName == "FP":
                FPGrowth_m()

            if param == "-rt" and AlgoName == "AP":
                Apriori_rt()

            if param == "-rt" and AlgoName == "FP":
                FPGrowth_rt()

            if param == "-pf":
                patternFileName = sys.argv[i+1]

                if patternFileName == "PatternForFP.txt":
                    file_path = 'C:\\Users\\Shabrina Shara\\PycharmProjects\\pythonProject5\\PatternForFP.txt'
                    sys.stdout = open(file_path, "a")
                    FPGrowth()
                    print()
                    print("These are the generated patterns from FP-Growth algorithm")

                elif patternFileName == "PatternForAP.txt":
                    file_path = 'C:\\Users\\Shabrina Shara\\PycharmProjects\\pythonProject5\\PatternForAP.txt'
                    sys.stdout = open(file_path, "a")
                    Apriori()
                    print("These are the generated patterns from Apriori algorithm")

            if param == "-n" and AlgoName == "AP":
                Apriori_n()

            if param == "-n" and AlgoName == "FP":
                FPGrowth_n()

            if param == "-pc" and AlgoName == "AP":
                Apriori()

            if param == "-pc" and AlgoName == "FP":
                FPGrowth()

            if param == "-o":
                outputFileName = sys.argv[i+1]

                if outputFileName == "Apriori.csv":
                    file_path = 'C:\\Users\\Shabrina Shara\\PycharmProjects\\pythonProject5\\Apriori.csv'
                    sys.stdout = open(file_path, "a"  )
                    Apriori_o()

                elif outputFileName == "FP.csv":
                    file_path = 'C:\\Users\\Shabrina Shara\\PycharmProjects\\pythonProject5\\FP.csv'
                    sys.stdout = open(file_path, "a")
                    FPGrowth_o()






#terminal input

#Python FPM.py -a FP -d mushroom.txt -t 0.5 -rt -m
#Python FPM.py -a AP -d mushroom.txt -t 0.5 -rt -m

#Python FPM.py -a AP -d Toy.txt -t 0.4 -rt -m
#Python FPM.py -a FP -d Toy.txt -t 0.4 -rt -m

#for check -n
#Python FPM.py -a AP -d Toy.txt -t 0.4 -n

#for check -pf
#Python FPM.py -a AP -d Toy.txt -t 0.4 -pf PatternForAP.txt

#for check -o
#Python FPM.py -a AP -d Toy.txt -t 0.4 -o Apriori.csv
#Python FPM.py -a FP -d Toy.txt -t 0.4 -o FP.csv







