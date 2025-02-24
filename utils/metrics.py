def accuracy(pred, gt):
    return (pred == gt).sum() / len(pred)