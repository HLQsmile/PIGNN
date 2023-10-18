import numpy as np
import pandas as pd
import argparse
from scipy.io import mmread
from model import (scGraph, edge_transform_func)


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()  # 添加命令行解析器，通俗一点可以理解为这个是添加参数并且管理参数的一个容器；
    parser.add_argument('-expr', '--expr', type=str, default=r'..\exp.csv')
    parser.add_argument('-label', '--label', type=str, default=r'..\labels.csv')
    parser.add_argument('-net', '--net', type=str, default=r'...\STRINGDB.graph.csv')
    parser.add_argument('-out', '--outfile', type=str, default=r'..\dataset.npz')
    parser.add_argument('-q', '--quantile', type=float, default='0.99')
    return parser

def map_graph_genes_to_expression_indices(graph_genes, expression_genes):
    '''
    convert graph_gene to index which is the order of converted_expr_gene and drop nan
    '''
    edge_dataframe = pd.DataFrame(graph_genes, columns=['source_gene', 'target_gene'])
    gene_to_index_mapping = {gene: index for index, gene in enumerate(expression_genes)}
    mapping_function = lambda x: gene_to_index_mapping.get(x, np.nan)
    edge_dataframe = edge_dataframe.applymap(mapping_function)
    edge_dataframe = edge_dataframe.dropna(axis='index')

    assert 'start_node' in edge_dataframe.columns
    assert 'end_node' in edge_dataframe.columns
    edge_index = edge_dataframe[['start_node', 'end_node']].T.values
    start_nodes, end_nodes = edge_index[0], edge_index[1]
    num_nodes_total = len(expression_genes) if len(expression_genes) is not None else np.max(edge_index) + 1

    self_loop_mask = start_nodes == end_nodes
    nodes_to_add = list(set(np.arange(0, num_nodes_total, dtype=int)) - set(start_nodes[self_loop_mask]))

    new_edges_df = pd.DataFrame()
    new_edges_df['start_node'] = nodes_to_add
    new_edges_df['end_node'] = nodes_to_add
    weight_column = 'strength'

    if weight_column in edge_dataframe.columns:
        new_edges_df[weight_column] = 1

    updated_edges_df = edge_dataframe.append(new_edges_df, ignore_index=True)

    return updated_edges_df.values


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:',args)

    expr_file = args.expr
    label_file = args.label
    net_file = args.net
    thres = args.quantile
    save_file = args.outfile
    assert 0<=thres<=1,"quantile should be a float value in [0,1]."

    data_df = pd.read_csv(expr_file,header=0,index_col=0)
    label_df = pd.read_csv(label_file,header=0,index_col=0)

    graph_df = pd.read_csv(net_file,header=None,index_col=None,)
    graph_df.columns = ['node1', 'node2', 'score']
    graph_df = graph_df.loc[graph_df.score.ge(graph_df.score.quantile(0.99)).values,['node1','node2']] # quantile 0.99

    # normalize + log1p transform for read counts
    data_df = data_df.apply(lambda x: 1e6 * x / x.sum() + 1e-5, axis=0)
    data_df = data_df.applymap(np.log1p)

    str_labels = np.unique(label_df.values).tolist()
    label = [str_labels.index(x) for x in label_df.values]
    gene = data_df.index.values
    barcode = data_df.columns.values
    edge_index =  map_graph_genes_to_expression_indices(graph_df.values, gene)

    print('shape of expression matrix [#genes,#cells]:', data_df.shape)
    print('shape of cell labels:', len(label))
    print('number of cell types:', len(str_labels))
    print('shape of backbone network:', edge_index.shape)

    data_dict = {}
    data_dict['barcode'] = barcode
    data_dict['gene'] = gene
    data_dict['logExpr'] = np.nan_to_num(data_df.values)
    data_dict['str_labels'] = str_labels
    data_dict['label'] = label
    data_dict['edge_index'] = edge_index

    np.savez(save_file, **data_dict)

    print('Finished.')
    print(np.isnan(data_dict['logExpr']).any())