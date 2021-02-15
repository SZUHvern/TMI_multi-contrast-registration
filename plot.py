import numpy as np

class cal_seg():
    def __init__(self, y_true, y_pred):
        self.y_pred = np.round(np.clip(y_pred, 0, 1))
        self.y_true = np.round(np.clip(y_true, 0, 1))

        self.tp = self.TP()
        self.fp = self.FP()
        self.tn = self.TN()
        self.fn = self.FN()

    def TP(self):
        true_positives = np.sum(np.round(np.clip(self.y_true * self.y_pred, 0, 1)))
        return true_positives

    def FP(self):
        y_pred_f01 = np.sum(np.round(np.clip(self.y_pred, 0, 1)))
        false_positives = y_pred_f01 - self.TP()
        return false_positives

    def TN(self):
        y_pred_f01 = np.round(np.clip(self.y_pred, 0, 1))
        all_one = np.ones_like(y_pred_f01)
        y_pred_f_1 = -1 * (y_pred_f01 - all_one)
        y_true_f_1 = -1 * (self.y_true - all_one)
        true_negatives = np.sum(np.round(np.clip(y_true_f_1 + y_pred_f_1, 0, 1)))

        return true_negatives

    def FN(self):
        tp_f01 = np.round(np.clip(self.y_true * self.y_pred, 0, 1))
        false_negatives = np.sum(np.round(np.clip(self.y_true - tp_f01, 0, 1)))
        return false_negatives

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def dice(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

def grid(val_F):
    grid = np.zeros((224, 224, 1))
    for i in range(224):
        for j in range(224):
            if i % 16 < 1:
                grid[i][j][0] = 1
            if j % 16 < 1:
                grid[i][j][0] = 1
    grid_array = np.ones_like(val_F)
    for i in range(grid_array.shape[0]):
        grid_array[i] = grid * grid_array[i]
    return grid_array
