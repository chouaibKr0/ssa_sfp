from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,matthews_corrcoef

class SFPMetrics:
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_prob=None):
        metrics = {
            'mcc': matthews_corrcoef(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            
        return metrics
