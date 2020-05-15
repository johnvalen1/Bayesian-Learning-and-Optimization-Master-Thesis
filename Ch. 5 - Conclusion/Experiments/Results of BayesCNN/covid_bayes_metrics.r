
dir = 'C:/Users/John/Desktop/Thesis/Ch. 5 - Conclusion/'
setwd(dir)
covid_Bayes_results = read.csv('result_dataframe_for_30_epochs.csv')

attach(covid_Bayes_results)
names(covid_Bayes_results)
# distribution-free confidence intervals on train accuracy, test accuracy, gg, uncertainties,



# -------------------- DISTRIBUTION-FREE CONFIDENCE INTERVALS -------------------------------
quantile(Generalization.Gap, c(.025, .975))
median(Generalization.Gap)

quantile(Train.Accuracy, c(.025, .975))
median(Train.Accuracy)

# Test accuracy is constant. No need for interval.

quantile(Avg..Epistemic, c(.025, .975))
median(Avg..Epistemic)

quantile(Avg..Aleatoric, c(.025, .975))
median(Avg..Aleatoric)

quantile(Occam.factor..FC., c(.025, .975))
median(Occam.factor..FC.)

quantile(Occam.factor..Filter., c(.025, .975))
median(Occam.factor..Filter.)

# We can average the Occam factors (one is based on fully-connected weight, the other on a filter weight)
Avg_Occam_Factor = rowMeans(cbind(Occam.factor..FC., Occam.factor..Filter.))

quantile(Avg_Occam_Factor, c(.025, .975))
median(Avg_Occam_Factor)

quantile(Train.Loss, c(.025, .975))
median(Train.Loss)

quantile(Test.Loss, c(.025, .975))
median(Test.Loss)


# define confusion matrix
confusion = read.csv('confusion_matrix_bayes.csv')
confusion

TP = confusion[1,2]
TP
FP = confusion[1,3]
FP
FN = confusion[2,2]
FN
TN = confusion[2,3]
TN

sensitivity = TP/(TP+FN)
specificity = TN/(FP+TN)

sensitivity
specificity

