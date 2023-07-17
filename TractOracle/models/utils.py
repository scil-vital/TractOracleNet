def calc_accuracy(y, y_hat, threshold=0.5):
    return ((y_hat > threshold).int() == y.int()).float().mean()
