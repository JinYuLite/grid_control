'''
用来生成配置赛题的yaml文件
主要的思路为：从example处得到动态参数，加上静态参数，合并为
main.yml的整体yaml文件。
运行main.py时，从utilize/main.yml文件处读取信息
'''
import os
import re

import yaml
import pandas as pd
import example


def _round(x):
    return round(x, 2)


def read_grid_data():
    grid = example.Print()

    dataset_path = args.dataset_path

    load_p_filepath = os.path.join(dataset_path, 'load_p.csv')
    load_q_filepath = os.path.join(dataset_path, 'load_q.csv')
    gen_p_filepath = os.path.join(dataset_path, 'gen_p.csv')
    gen_q_filepath = os.path.join(dataset_path, 'gen_q.csv')
    max_renewable_gen_p_filepath = os.path.join(dataset_path, 'max_renewable_gen_p.csv')
    keep_decimal_digits = 2
    assert os.path.isfile(load_p_filepath), "Cannot find the data file."
    assert os.path.isfile(gen_p_filepath), "Cannot find the data file."

    grid.readdata(1, load_p_filepath, load_q_filepath, gen_p_filepath, gen_q_filepath)
    grid.env_feedback(grid.name_unp, grid.itime_unp[0], [], 1, [])

    # 机组名称
    lnname = grid.lnname

    # 不同类型机组机组编号
    gen_type = grid.gen_type
    thermal_ids = [i for i, x in enumerate(grid.gen_type) if x == 1]
    renewable_ids = [i for i, x in enumerate(grid.gen_type) if x == 5]
    balanced_ids = [i for i, x in enumerate(grid.gen_type) if x == 2]
    balanced_id = balanced_ids[0]


    # 机组有功出力上下限
    max_gen_p = [_round(x) for x in grid.gen_plimit]
    min_gen_p = [_round(x) for x in grid.gen_pmin]

    # 机组有功出力上下限
    max_gen_q = [_round(x) for x in grid.gen_qmax]
    min_gen_q = [_round(x) for x in grid.gen_qmin]

    # 机组电压上下限
    max_gen_v = [_round(x) for x in grid.gen_vmax]
    min_gen_v = [_round(x) for x in grid.gen_vmin]

    # 节点电压上下限
    max_bus_v = [_round(x) for x in grid.bus_vmax]
    min_bus_v = [_round(x) for x in grid.bus_vmin]

    # 线路电流热极限
    line_thermal_limit = [_round(x) for x in grid.line_thermal_limit]

    # 断面数、机组数、支路数
    num_sample = pd.read_csv(load_p_filepath).shape[0]
    num_gen = len(grid.unname)
    num_line = len(grid.lnname)

    un_nameindex_key = list(grid.un_nameindex.keys())
    un_nameindex_value = list(grid.un_nameindex.values())



    dict_ = {
        'lnname': lnname,
        'max_gen_p': max_gen_p,
        'min_gen_p': min_gen_p,
        'max_gen_q': max_gen_q,
        'min_gen_q': min_gen_q,
        'max_gen_v': max_gen_v,
        'min_gen_v': min_gen_v,
        'max_bus_v': max_bus_v,
        'min_bus_v': min_bus_v,
        'num_gen': num_gen,
        'num_line': num_line,
        'renewable_ids': renewable_ids,
        'thermal_ids': thermal_ids,
        'balanced_id': balanced_id,
        'gen_type': gen_type,
        'num_sample': num_sample,
        'load_p_filepath': load_p_filepath,
        'load_q_filepath': load_q_filepath,
        'gen_p_filepath': gen_p_filepath,
        'gen_q_filepath': gen_q_filepath,
        'max_renewable_gen_p_filepath':  max_renewable_gen_p_filepath,
        'line_thermal_limit': line_thermal_limit,
        'un_nameindex_key': un_nameindex_key,
        'un_nameindex_value': un_nameindex_value,
    }
    return dict_


# 需要读取grid信息才能得到的yaml称为dynamic
def create_dynamic_yml():
    dict_ = read_grid_data()
    with open('utilize/parameters/dynamic.yml', 'w+', encoding='utf-8') as f:
        for key, val in dict_.items():
            stream = yaml.dump({key: val}, default_flow_style=True)
            # 整理格式，方法比较笨，要求不能有nested dict
            f.write(re.sub(r'{|}', '', stream))


def merge_dynamic_static_yml():
    with open('utilize/parameters/dynamic.yml', 'r', encoding='utf-8') as f:
        dict_dynamic = yaml.load(f, Loader=yaml.Loader)
    with open('utilize/parameters/static.yml', 'r', encoding='utf-8') as f:
        dict_static = yaml.load(f, Loader=yaml.Loader)

    dict_static.update(dict_dynamic)
    with open('utilize/parameters/main.yml', 'w+', encoding='utf-8') as f:
        for key, val in dict_static.items():
            stream = yaml.dump({key: val}, default_flow_style=True)
            # 整理格式，方法比较笨，要求不能有nested dict
            f.write(re.sub(r'{|}', '', stream))


def main():
    create_dynamic_yml()
    merge_dynamic_static_yml()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="data", type=str, help='The path of the dataset.')
    args = parser.parse_args()
    
    main()
