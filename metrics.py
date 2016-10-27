
from sklearn.metrics import accuracy_score,precision_score,recall_score

def metrics(y, y_hat):
	print "accuracy:", accuracy_score(y, y_hat)
	print "precision:", precision_score(y, y_hat)
	print "recall", recall_score(y, y_hat)