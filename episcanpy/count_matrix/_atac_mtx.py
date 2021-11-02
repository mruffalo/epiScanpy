from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import anndata
import anndata as ad
import bamnostic as bs
import numpy as np
import pandas as pd
import scipy.sparse
from bx.intervals import Interval, IntervalTree

from ..utils import removeprefix

MOUSE = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
        '2', '3', '4', '5', '6', '7', '8', '9','X', 'Y']
HUMAN = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
        '2', '3', '4', '5', '6', '7', '8', '9','X', 'Y']

def get_barcode_from_read(bam_file: Path, read: bs.AlignedSegment, barcode_tag: str = 'CB'):
    return read.get_tag(barcode_tag)

def get_barcode_from_bam_filename(bam_file: Path, read: bs.AlignedSegment, barcode_tag: str):
    return bam_file.stem

def get_feature_df(chromosomes: List[str], loaded_feat: Dict[str, List[List[int]]]) -> pd.DataFrame:
    feature_dfs = []
    for chrom in chromosomes:
        features = loaded_feat[chrom]
        chroms = [chrom] * len(features)
        index = []
        starts = []
        ends = []
        for start, end in features:
            index.append(f'{chrom}:{start}-{end}')
            starts.append(start)
            ends.append(end)
        feature_dfs.append(pd.DataFrame({'chrom': chroms, 'start': starts, 'end': ends}, index=index))
    return pd.concat(feature_dfs)

def bld_atac_mtx(
    bam_files: List[Path],
    loaded_feat: Dict[str, List[List[int]]],
    output_file_path: Optional[Path] = None,
    check_sq=True,
    chromosomes=HUMAN,
    cb_tag='CB',
    barcode_func: Callable[[Path, bs.AlignedSegment, str], str] = get_barcode_from_bam_filename,
) -> anndata.AnnData:
    """
    Build a count matrix one set of features at a time. It is specific of ATAC-seq data.
    It curently do not write down a sparse matrix. It writes down a regular count matrix
    as a text file.

    Parameters
    ----------

    list_bam_files: input must be a list of bam file names. One for each cell to
        build the count matrix for

    loaded_feat: the features for which you want to build the count matrix

    output_file_name: name of the output file. The count matrix that will be written
        down in the current directory. If this parameter is not specified,
        the output count amtrix will be named 'std_output_ct_mtx.txt'

    path: path where to find the input file. The output file will be written down
    in your current directory, it is not affected by this parameter.

    writing_option: standard writing options for the output file. 'a' or 'w'
        'a' to append to an already existing matrix. 'w' to overwrite any
        previously exisiting matrix.
        default: 'a'

    header: if you want to write down the feature name specify this argument.
        Input must be a list.

    mode: bamnostic argument 'r' or 'w' for read and write 'b' and 's' for bam or sam
        if only 'r' is specified, bamnostic will try to determine if the input is
        either a bam or sam file.

    check_sq: bamnostic argument. when reading, check if SQ entries are present in header

    chromosomes: chromosomes of the species you are considering. default value
        is the human genome (not including mitochondrial genome).
        HUMAN = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
                '2', '3', '4', '5', '6', '7', '8', '9','X', 'Y']
        MOUSE = '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '2', '3', '4', '5', '6', '7', '8', '9','X', 'Y']

    Return
    ------
    It does not return any object. The function write down the desired count
    matrix in a txt file

    """

    if output_file_path is None:
        output_file_path = Path('std_output_ct_mtx.h5ad')

    chrom_overlap = set(chromosomes) & set(loaded_feat)
    feature_df = get_feature_df(chromosomes, loaded_feat)

    # Maps chromosome names to interval trees, with tree values as indexes
    # into the feature list for each chromosome; these indexes are used to
    # count reads falling into each feature
    trees: Dict[str, IntervalTree] = {}
    for chrom in chrom_overlap:
        tree = IntervalTree()
        for i, (start, stop) in enumerate(loaded_feat[chrom]):
            tree.insert_interval(Interval(start, stop, value=i))
        trees[chrom] = tree

    def get_feature_vectors():
        return {key: np.zeros(len(loaded_feat[key]), dtype=int) for key in chromosomes}

    cell_feature_counts = defaultdict(get_feature_vectors)

    # start going through the bam files
    for bam_file in bam_files:
        samfile = bs.AlignmentFile(bam_file, mode="rb", check_sq=check_sq)
        for read in samfile:
            chrom = removeprefix(read.reference_name, 'chr')
            if chrom not in trees:
                continue
            start, end = read.pos, read.pos + read.query_length
            barcode = barcode_func(bam_file, read, cb_tag)
            chrom_counts = cell_feature_counts[barcode]
            for interval in trees[chrom].find(start, end):
                chrom_counts[chrom][interval.value] += 1

    cells = []
    cell_vecs = []
    for cell, feature_counts in cell_feature_counts.items():
        # Dict ordering is stable, but be explicit in matching the ordering in feature_df
        cell_vec_chunks = []
        for chrom in chromosomes:
            cell_vec_chunks.append(feature_counts[chrom])
        cell_vec = scipy.sparse.hstack(cell_vec_chunks)
        cells.append(cell)
        cell_vecs.append(cell_vec)

    cell_mat = scipy.sparse.csr_matrix(scipy.sparse.vstack(cell_vecs))

    adata = anndata.AnnData(
        X=cell_mat,
        obs=pd.DataFrame(index=cells),
        var=feature_df,
        dtype=int,
    )
    adata = adata[sorted(cells), :].copy()
    adata.write_h5ad(output_file_path)

    return adata

def read_mtx_bed(file_name, path='', omic='ATAC'):
    """
    read this specific matrix format. It is the standard output of bedtools when you merge bam files.
    """
    peak_name = []
    cell_matrix = []
    with open(path+file_name) as f:
        head = f.readline().split('\t')
        head[len(head)-1] = head[len(head)-1].split("\n")[0]
        for line in f:
            line = line.split('\t')
            line[len(line)-1] = line[len(line)-1].split("\n")[0]
            peak_name.append(line[3]) # for some reason it has rownames
            cell_matrix.append([int(x) for x in line[4:]])
    cell_names = head[4:]
    cell_matrix=np.matrix(cell_matrix)
    cell_matrix = cell_matrix.transpose()
    adata = ad.AnnData(cell_matrix,
                   obs=pd.DataFrame(index=cell_names),
                   var=pd.DataFrame(index=peak_name))
    if omic is not None:
        adata.uns['omic'] = omic
    return(adata)


def save_sparse_mtx(initial_matrix, output_file='.h5ad', path='', omic='ATAC', bed=False, save=True):
    """
    Convert regular atac matrix into a sparse Anndata:

    Parameters
    ----------

    initial_matrix: initial dense count matrix to load and convert into a sparse matrix.
    If bed = True,  initial_matrix should be the path to the bed file.

    output_file: name of the output file for the AnnData object.
    Default output is the name of the input file with .h5ad extension

    path: path to the input count matrix. The AnnData object is written in the current directory,
        not the location specified in path.

    omic: 'ATAC', 'RNA' or 'methylation' are the 3 currently recognised omics in epiScanpy.
        However, other omic name can be accepted but are not yet recognised in other functions.
        default: 'ATAC'

    bed: boolean. If True it consider another input format (bedtools output format for count matrices)

    save: boolean. If True, the sparse matrix is saved as h5ad file. Otherwise it is simply return.

    Return
    ------

    It returns the loaded matrix as an AnnData object.
    """
    head = None
    data = []
    cell_names = []

    # choice between 2 different input count matrix formats
    if bed:
        adata = read_mtx_bed(initial_matrix, path, omic)
    else:
        # reading the non sparse file
        with open(path+initial_matrix) as f:
            first_line = f.readline()
            first_line = first_line[:-3].split('\t')
            if first_line[0] == 'sample_name':
                new_line = []
                for value in first_line:
                    if value != '':
                        new_line.append(value)
                head = new_line.copy()
            else:
                cell_names.append(first_line[0])
                data = [[int(l) for l in first_line[1:-1]]]
            file = f.readlines()

        for line in file:
            line = line[:-3].split('\t')
            cell_names.append(line[0])
            new_line = []
            for value in line[1:]:
                if value != '':
                    new_line.append(value)
            data.append([int(l) for l in new_line])


        anndata_kwargs = {}
        # convert into an AnnData object
        if head is not None:
            anndata_kwargs['var'] = pd.DataFrame(index=head[1:])
        adata = ad.AnnData(scipy.sparse.csr_matrix(data), obs=pd.DataFrame(index=cell_names), **anndata_kwargs)

        if omic is not None:
            adata.uns['omic'] = omic

    # writing the file as h5ad --> sparse matrix with minimum annotations
    if save:
        if output_file=='.h5ad':
            output_file = "".join([initial_matrix.split('.')[0], output_file])

        adata.write(output_file)

    return(adata)
