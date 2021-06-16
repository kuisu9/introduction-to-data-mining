import pandas as pd

def delete_unknown(data):
    for column in list(data.columns):
        data = data[data[column] != "unknown"]
    return data


def redefine(data):
    col_number = 0
    for column in list(data.columns):
        redef_col = list(data[column])
        for row in range(len(redef_col)):
            redef_col[row] = str(col_number) + "_" + redef_col[row]
        data[column] = redef_col
        col_number += 1
    return data


# 3. 从候选项集中选出频繁项集
# 如下图所以我们需要从初始的候选项集中计算k项频繁项集，所以这里封装函数用于每次计算频繁项集及支持度，
# 当候选项集中集合中的每个元素都存在事务记录集合中是计数并保存到字典中，计算支持度后输出频繁项集和支持度。
def generate_freq_supports(data_set, item_set, min_support):
    freq_set = set()  # 保存频繁项集元素
    item_count = {}  # 保存元素频次，用于计算支持度
    supports = {}  # 保存支持度

    # 如果项集中元素在数据集中则计数
    for record in data_set:
        for item in item_set:
            if item.issubset(record):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

    data_len = float(len(data_set))

    # 计算项集支持度
    for item in item_count:
        if (item_count[item] / data_len) >= min_support:
            freq_set.add(item)
            supports[item] = item_count[item] / data_len

    return freq_set, supports

# 4.生成新组合
# 由初始候选集会生成{1,2,3,5}的频繁项集，后续需要生成新的候选项集Ck。
def generate_new_combinations(freq_set, k):
    new_combinations = set()  # 保存新组合
    sets_len = len(freq_set)  # 集合含有元素个数，用于遍历求得组合
    freq_set_list = list(freq_set)  # 集合转为列表用于索引

    for i in range(sets_len):
        for j in range(i + 1, sets_len):
            l1 = list(freq_set_list[i])
            l2 = list(freq_set_list[j])
            l1.sort()
            l2.sort()

            # 项集若有相同的父集则合并项集
            if l1[0:k-2] == l2[0:k-2]:
                freq_item = freq_set_list[i] | freq_set_list[j]
                new_combinations.add(freq_item)

    return new_combinations

# 5.循环生成候选集集频繁集

def apriori(data_set, min_support, max_len=None):
    max_items = 2  # 初始项集元素个数
    freq_sets = []  # 保存所有频繁项集
    supports = {}  # 保存所有支持度

    # 候选项1项集
    c1 = set()
    for items in data_set:
        for item in items:
            item_set = frozenset([item])
            c1.add(item_set)

    # 频繁项1项集及其支持度
    l1, support1 = generate_freq_supports(data_set, c1, min_support)

    freq_sets.append(l1)
    supports.update(support1)

    if max_len is None:
        max_len = float('inf')

    while max_items and max_items <= 3:
        ci = generate_new_combinations(freq_sets[-1], max_items)  # 生成候选集
        li, support = generate_freq_supports(data_set, ci, min_support)  # 生成频繁项集和支持度

        # 如果有频繁项集则进入下个循环
        if li:
            freq_sets.append(li)
            supports.update(support)
            max_items += 1
        else:
            max_items = 0

    return freq_sets, supports

# 6.生成关联规则
def association_rules(freq_sets, supports, min_conf):
    rules = []
    max_len = len(freq_sets)

    # 生成关联规则，筛选符合规则的频繁集计算置信度，满足最小置信度的关联规则添加到列表
    for k in range(max_len - 1):
        for freq_set in freq_sets[k]:
            for sub_set in freq_sets[k + 1]:
                if freq_set.issubset(sub_set):
                    conf = supports[sub_set] / supports[freq_set]
                    rule = (freq_set, sub_set - freq_set, conf)
                    if conf >= min_conf:
                        if frozenset(["20_no"]).issubset(sub_set - freq_set) or frozenset(["20_yes"]).issubset(sub_set - freq_set):
                            rules.append(rule)
    return rules

# 7.主程序
if __name__ == '__main__':

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    for i in data.columns:
        if type(data[i][0]) is str:
            print("unknown value count in "+i+":\t" + str(data[data[i] == 'unknown']['y'].count()))
    # 预处理
    data = delete_unknown(data)
    data = data.astype(str)
    data = redefine(data)
    data = data.values.tolist()

    L, support_data = apriori(data, min_support=0.5)
    association_rules = association_rules(L, support_data, min_conf=0.8)
    association_rules.sort(key=lambda x: x[-1], reverse=True)
    for rule in association_rules:
        print(rule)