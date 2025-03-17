class Metrics:
    def __init__(self):
        self.train_accuracy = None
        self.train_f1 = None
        self.val_accuracy = None
        self.val_f1 = None

    def __repr__(self):
        ret = ''
        ret += 'Training: Accuracy={:.3f}, F1={:.3f}\n'.format(self.train_accuracy, self.train_f1)
        ret += 'Validation: Accuracy={:.3f}, F1={:.3f}\n'.format(self.val_accuracy, self.val_f1)
        return ret
