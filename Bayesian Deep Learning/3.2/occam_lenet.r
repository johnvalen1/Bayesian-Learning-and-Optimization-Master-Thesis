# Examining a relationship between the performance metrics and the Occam factor
# of a Bayesian LeNet-5.
directory = "C:/Users/John/Desktop/Thesis/Ch. 3 -Bayesian deep learning/Experiments/3.2/"
setwd(directory)
results = read.csv('result_dataframe.csv', header = TRUE)
#install.packages("corrplot")
library(corrplot)
attach(results)
names(results)

results = results[-c(1)] #exclude model number

File <- "./img/corr_matrix.png"
png(File)
dir.create(dirname(File), showWarnings = FALSE)
cor_matrix = cor(results)
cor_matrix
corrplot(cor_matrix, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
corrplot(cor_matrix, method="number")
dev.off()


File <- "./img/corr_matrix_numbers.png"
png(File)
dir.create(dirname(File), showWarnings = FALSE)
cor_matrix = cor(results)
cor_matrix
corrplot(cor_matrix, method="number")
dev.off()

#scatter plots... to check if any relationships are NOT linear:

File <- "./img/avg_aleatoric_vs_avg_epistemic.png"
png(File)
plot(Avg..Aleatoric, Avg..Epistemic, main="Avg. Aleatoric vs. Avg. Epistemic",
     xlab="Avg. Aleatoric ", ylab="Avg. Epistemic", pch=19)
dev.off()


File <- "./img/FC_occam_vs_avg_epistemic.png"
png(File)
plot(Occam.factor..FC., Avg..Epistemic, main="FC Occam Factor vs. Avg. Epistemic",
     xlab="Occam Factor based on FC weight", ylab="Avg. Epistemic", pch=19)
dev.off()

File <- "./img/FC_occam_vs_avg_aleatoric.png"
png(File)
plot(Occam.factor..FC., Avg..Aleatoric, main="FC Occam Factor vs. Avg. Aleatoric",
     xlab="Occam Factor based on FC weight", ylab="Avg. Aleatoric", pch=19)
dev.off()


File <- "./img/FC_occam_vs_test_accuracy.png"
png(File)
plot(Occam.factor..FC., Test.Accuracy, main="FC Occam Factor vs. Test Accuracy",
     xlab="Occam Factor based on FC weight ", ylab="Test Accuracy", pch=19)
dev.off()

File <- "./img/Filter_occam_vs_test_accuracy.png"
png(File)
plot(Occam.factor..Filter., Test.Accuracy, main="Filter Occam Factor vs. Test Accuracy",
     xlab="Occam Factor based on Filter weight ", ylab="Test Accuracy", pch=19)
dev.off()

File <- "./img/Filter_occam_vs_FC_occam.png"
png(File)
plot(Occam.factor..Filter., Occam.factor..FC., main="Filter Occam Factor vs. Occam Factor based on FC weight",
     xlab="Occam Factor based on filter weight ", ylab="Occam Factor based on FC weight", pch=19)
dev.off()





#File <- "./img/FC_occam_vs_test_accuracy.png"
#png(File)
#plot(Occam.factor..FC., Test.Accuracy, main="FC Occam Factor vs. Test Accuracy",
#     xlab="Occam Factor based on FC weight ", ylab="Test Accuracy", pch=19)
#dev.off()

#Finish writing all scatterplots!


