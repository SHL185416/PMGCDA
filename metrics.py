import numpy as np
from sklearn.metrics import f1_score


def f1_scores(y_pred, y_true):
    r"""calculate f1 score。

    Calculate the f1 score of the model。

    :arg
        y_pred: the predicted labels.
        y_true: the true labels.
    :return
        f1_score: the f1 score of the model.
    """

    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]
