import numpy as np

def Accuracy(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the accuracy for prediction.
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - Acc : (scalar, float), Computed accuracy score
    # ========================= EDIT HERE =========================
    Acc = None

    if len(pred.shape) == 1:
       pred = np.expand_dims(pred, 1)
    if len(label.shape) == 1:
        label = np.expand_dims(label, 1)

    TP = 0
    TN = 0

    for i in range(label.shape[0]):
        if(label[i][0] == 1 and pred[i][0] == 1):
            TP += 1
        elif(label[i][0] == 0 and pred[i][0] == 0):
            TN += 1

    Acc = (TP + TN)/ label.shape[0]
    # =============================================================
    return Acc

def Precision(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Precision for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - precision : (scalar, float), Computed precision score
    # ========================= EDIT HERE =========================
    precision = None
    if len(pred.shape) == 1:
       pred = np.expand_dims(pred, 1)
    if len(label.shape) == 1:
        label = np.expand_dims(label, 1)

    TP = 0
    FP = 0

    for i in range(label.shape[0]):
        if(label[i][0] == 1 and pred[i][0] == 1):
            TP += 1
        elif(label[i][0] == 0 and pred[i][0] == 1):
            FP += 1
    
    precision = TP / (TP + FP)
    # =============================================================
    return precision

def Recall(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the Recall for prediction.
    #         you should consider that label = 1 is positive. 0 is negative
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - recall : (scalar, float), Computed recall score
    # ========================= EDIT HERE =========================
    recall = None

    if len(pred.shape) == 1:
       pred = np.expand_dims(pred, 1)
    if len(label.shape) == 1:
        label = np.expand_dims(label, 1)

    TP = 0
    FN = 0

    for i in range(label.shape[0]):
        if(label[i][0] == 1 and pred[i][0] == 1):
            TP += 1
        elif(label[i][0] == 1 and pred[i][0] == 0):
            FN += 1

    recall = TP / (TP+FN)
    # =============================================================
    return recall

def F_measure(label, pred):
    ########################################################################################
    # TODO : Complete the code to calculate the F-measure score for prediction.
    #         you can erase the code. (F_score = 0.)
    #         Notice that, if you encounter the divide zero, return 1
    #         [Input]
    #         - label : (N, ), Correct label with 0 (negative) or 1 (positive)
    #         - hypo  : (N, ), Predicted score between 0 and 1
    #         [output]
    #         - F_score : (scalar, float), Computed F-score score
    # ========================= EDIT HERE =========================
    F_score = None
    if len(pred.shape) == 1:
       pred = np.expand_dims(pred, 1)
    if len(label.shape) == 1:
        label = np.expand_dims(label, 1)

    TP = 0
    FP = 0
    FN = 0

    for i in range(label.shape[0]):
        if(label[i][0] == 1 and pred[i][0] == 1):
            TP += 1
        elif(label[i][0] == 0 and pred[i][0] == 1):
            FP += 1
        elif(label[i][0] == 1 and pred[i][0] == 0):
            FN += 1

    p = TP/(TP + FP)
    r = TP/(TP + FN)
    F_score = (2*p*r) / (p+r)

    # =============================================================
    return F_score

def MAP(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the code to calculate the MAP for prediction.
    #         Notice that, hypo is the real value array in (0, 1)
    #         MAP (at = 10) means MAP @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed MAP score
    # ========================= EDIT HERE =========================
    Map = None
    argsorted_hypo = np.flip(np.argsort(hypo),1)
    sorted_label = np.zeros(label.shape)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            k = argsorted_hypo[i][j]
            sorted_label[i][j] = label[i][k]

    precision_np = np.zeros(label.shape)
    ap = []
    for i in range(label.shape[0]):
        rel_sum = 0
        for j in range(label.shape[1]):
            if sorted_label[i][j] == 1:
                rel_sum += 1
                precision_np[i][j] = rel_sum/(j+1)
        
        tmp_ap = 0
        for k in range(at):
            tmp_ap += precision_np[i][k]
        tmp_ap /= rel_sum
        ap.append(tmp_ap)
    
    Map = sum(ap) / label.shape[0]

    # =============================================================
    return Map

def nDCG(label, hypo, at = 10):
    ########################################################################################
    # TODO : Complete the each code to calculate the nDCG for prediction.
    #         you can erase the code. (dcg, idcg, ndcg = 0.)
    #         Notice that, hypo is the real value array in (0, 1)
    #         nDCG (at = 10 ) means nDCG @10
    #         [Input]
    #         - label : (N, K), Correct label with 0 (incorrect) or 1 (correct)
    #         - hypo  : (N, K), Predicted score between 0 and 1
    #         - at: (int), # of element to consider from the first. (TOP-@)
    #         [output]
    #         - Map : (scalar, float), Computed nDCG score



    def DCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        dcg = []
        argsorted_hypo = np.flip(np.argsort(hypo),1)
        sorted_label = np.zeros(label.shape)

        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                k = argsorted_hypo[i][j]
                sorted_label[i][j] = label[i][k]
        
        for i in range(label.shape[0]):
            tmp_dcg = 0
            for j in range(at):
                k = np.log2(j+2)
                if sorted_label[i][j] == 1:
                    tmp_dcg += 1/k
            dcg.append(tmp_dcg)


        # =============================================================
        return dcg

    def IDCG(label, hypo, at=10):
        # ========================= EDIT HERE =========================
        print(label.shape)
        idcg = []
        argsorted_hypo = np.flip(np.argsort(hypo),1)
        sorted_label = np.zeros(label.shape)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                k = argsorted_hypo[i][j]
                sorted_label[i][j] = label[i][k]
        
        idcg_label = -np.sort(-sorted_label)
        for i in range(label.shape[0]):
            tmp_idcg = 0
            for j in range(at):
                k = np.log2(j+2)
                if idcg_label[i][j] == 1:
                    tmp_idcg += 1/k
            idcg.append(tmp_idcg)    
        
        # =============================================================
        return idcg
    # ========================= EDIT HERE =========================
    ndcg = 0
    dcg_list = DCG(label, hypo, at)
    idcg_list = IDCG(label, hypo, at)
    dcg_len = len(dcg_list)
    tmp_ndcg = 0
    for i in range(dcg_len):
        tmp_ndcg += (dcg_list[i])/ (idcg_list[i])
    ndcg = tmp_ndcg / dcg_len
    # =============================================================
    return ndcg

# =============================================================== #
# ===================== DO NOT EDIT BELOW ======================= #
# =============================================================== #

def evaluation_test1(label, pred, at = 10):
    result = {}

    result['Accuracy '] = Accuracy(label, pred)
    result['Precision'] = Precision(label, pred)
    result['Recall   '] = Recall(label, pred)
    result['F_measure'] = F_measure(label, pred)

    return result

def evaluation_test2(label, hypo, at = 10):
    result = {}

    result['MAP  @%d'%at] = MAP(label, hypo, at)
    result['nDCG @%d'%at] = nDCG(label, hypo, at)

    return result
