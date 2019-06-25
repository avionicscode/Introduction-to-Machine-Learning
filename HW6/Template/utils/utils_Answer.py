import numpy as np

def Gini_index(Y_data):
    gini = 0
    #=========    Edit here    ==========
    #
    gini = 1
    distinct_result = np.unique(Y_data)

    for result in distinct_result:
        tmp_val = (Y_data == result).sum() / len(Y_data)
        gini -= (tmp_val**2)
    #
    #====================================
    return gini

def Entropy(Y_data):

    entropy = 0
    # =====    Edit here    ========
    #    
    distinct_result = np.unique(Y_data)

    for result in distinct_result:
        tmp_val = (Y_data == result).sum() / len(Y_data)
        entropy -= tmp_val*(np.log2(tmp_val))
    #
    # ==============================
    return entropy

def impurity_func(Y_data, criterion):

    if criterion == 'gini':
        return Gini_index(Y_data)

    elif criterion == 'entropy':
        return Entropy(Y_data)

def Finding_split_point(df, feature, criterion):

    col_data = df[feature]
    Y = df.values[:, -1]
    distinct_data = np.unique(col_data)

    split_point = distinct_data[0]
    min_purity = 1
    
    for idx, val in enumerate(distinct_data):
        less_idx = (col_data < val)

        y0 = Y[less_idx]
        y1 = Y[~less_idx]

        p0 = len(y0) / len(Y)
        p1 = len(y1) / len(Y)

        purity = np.sum([p0 * impurity_func(y0, criterion), p1 * impurity_func(y1, criterion)])

        if min_purity > purity:
            min_purity = purity
            split_point = val

    return split_point

def Gaussian_prob(x, mean, std):
    '''
    :param x: input value: X
    :param mean: the mean of X
    :param std: the standard deviation of X
    :return: probaility (X) ~ N(μ, σ^2)
    '''
    ret = 0
    # ========      Edit here         ==========
    exp = np.exp(-(pow(x-mean,2)/(2*pow(std,2))))
    ret = (1/ (np.sqrt(2*np.pi) * std)) * exp
    # =========================================
    return ret