
# GOAL:
# TO explore if choice of acquisition function / GP parameterization affects the significance of hyperparameters
# ... loss and accuracy are response metrics, but we should look at uncertainties as well if possible to
# incorporate.

setwd('C://Users//John//Desktop//Thesis//Ch. 4 - Bayesian Optimization//Experiments//My code//Without validation split//')
hyperparams_EI = read.csv('BayesOpt_Hyperparameters_keras_MNIST_EI.csv')
hyperparams_MPI = read.csv('BayesOpt_Hyperparameters_keras_MNIST_MPI.csv')
hyperparams_LCB = read.csv('BayesOpt_Hyperparameters_keras_MNIST_LCB.csv')

# EI:
attach(hyperparams_EI)
l1_drop = Dropout.in.layer.1
l2_drop = Dropout.in.layer.2
l1_out = Layer.1.size
l2_out = Layer.2.size
batch_size = Batch.size
epochs =  Epochs
EI_loss = Loss
EI_acc = Accuracy


# MPI:
attach(hyperparams_MPI)
l1_drop = Dropout.in.layer.1
l2_drop = Dropout.in.layer.2
l1_out = Layer.1.size
l2_out = Layer.2.size
batch_size = Batch.size
epochs =  Epochs
MPI_loss = Loss
MPI_acc = Accuracy


# LCB:
attach(hyperparams_LCB)
l1_drop = Dropout.in.layer.1
l2_drop = Dropout.in.layer.2
l1_out = Layer.1.size
l2_out = Layer.2.size
batch_size = Batch.size
epochs =  Epochs
LCB_loss = Loss
LCB_acc = Accuracy



#plotting the 'performance' curves for acquisition functions
#Accuracy
plot(EI_acc, col='red', cex = 0.5,main='Network test-set accuracy based on choice of acquisition', 
     type='l', ylab='Test-set accuracy',ylim=c(0.96,0.99), xlab = 'Number of BayesOpt iterations')
lines(MPI_acc, col = 'green')
lines(LCB_acc, col = 'blue')
legend('bottomright', legend=c("EI", "MPI", "LCB"),
       col=c("red", "green", "blue"), lty=1:2, cex=0.8)

#Loss
plot(EI_loss, col='red', cex = 0.5,main='Network test-set loss based on choice of acquisition', 
     type='l', ylab='Test-set loss',ylim=c(0.05,0.12), xlab = 'Number of BayesOpt iterations')
lines(MPI_loss, col = 'green')
lines(LCB_loss, col = 'blue')
legend('bottomright', legend=c("EI", "MPI", "LCB"),
       col=c("red", "green", "blue"), lty=1:2, cex=0.8)


#------------------------------------------ RANDOM SEARCH ------------------------------------------
setwd('C://Users//John//Desktop//Thesis//Ch. 4 - Bayesian Optimization//Experiments//My code//Random search//')
hyperparams_rs = read.csv('Random_Search_Hyperparameters_keras_MNIST.csv')

attach(hyperparams_rs)

l1_drop = Dropout.in.layer.1
l2_drop = Dropout.in.layer.2
l1_out = Layer.1.size
l2_out = Layer.2.size
batch_size = Batch.size
epochs =  Epochs
rs_loss = Loss
rs_acc = Accuracy



#plotting the 'performance' curves
#Accuracy
plot(rs_acc, col='red', cex = 0.5,main='Network test-set accuracy', 
     type='l', ylab='Test-set accuracy',ylim=c(0.95,0.99), xlab = 'Number of random search iterations')


#Loss
plot(rs_loss, col='red', cex = 0.5,main='Network test-set loss', 
     type='l', ylab='Test-set loss',ylim=c(0.05,0.14), xlab = 'Number of random search iterations')



# -------------------------------------- ANALYSIS/ COMPARISON -----------------------------------------

# At this point, one can verify if indeed BayesOpt is informed. For that, we would expect trend in the 'time series'
# of metrics. That is, BayesOpt accuracy should be a time series with positive trend, and BayesOpt loss a time series
# with negative trend. On the other hand, random search is uninformed, and should thus be white noise.

# ACCURACY
rs_acc_ts = ts(rs_acc)
ts.plot(rs_acc_ts)
max.lag=round(sqrt(length(rs_acc_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(rs_acc_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(rs_acc_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(rs_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(rs_acc_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(rs_acc_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(rs_acc_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(rs_acc_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)


# selected ARIMA model:
#ARIMA(1,1,1)
fit_ar1_ma1=arima(rs_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

# BayesOpt with EI Acquisition.....
EI_acc_ts = ts(EI_acc)
ts.plot(EI_acc_ts)
max.lag=round(sqrt(length(EI_acc_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(EI_acc_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(EI_acc_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(EI_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(EI_acc_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(EI_acc_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(EI_acc_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(EI_acc_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
# ARIMA(2,1,2)
fit_ar2_ma2=arima(EI_acc_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)


# BayesOpt with MPI Acquisition.....
MPI_acc_ts = ts(MPI_acc)
ts.plot(MPI_acc_ts)
max.lag=round(sqrt(length(MPI_acc_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(MPI_acc_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(MPI_acc_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(MPI_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(MPI_acc_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(MPI_acc_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(MPI_acc_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(MPI_acc_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
#ARIMA(2,1,1)
fit_ar2_ma1=arima(MPI_acc_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)



# BayesOpt with LCB Acquisition.....
LCB_acc_ts = ts(LCB_acc)
ts.plot(LCB_acc_ts)
max.lag=round(sqrt(length(LCB_acc_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(LCB_acc_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(LCB_acc_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(LCB_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(LCB_acc_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(LCB_acc_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(LCB_acc_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(LCB_acc_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
#ARIMA(1,1,1)
fit_ar1_ma1=arima(LCB_acc_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)




#test for stationarity
library(CADFtest)

CADFtest(rs_acc_ts, type= "drift", criterion= "BIC", max.lag.y=max.lag) # reject null of existence of unit root - stationary

CADFtest(EI_acc, type= "drift", criterion= "BIC", max.lag.y=max.lag) # fail to reject null, existence of unit root

CADFtest(LCB_acc, type= "drift", criterion= "BIC", max.lag.y=max.lag) # fail to reject null, existence of unit root

CADFtest(MPI_acc, type= "drift", criterion= "BIC", max.lag.y=max.lag) # reject null of existence of unit root - stationary




# LOSS
rs_loss_ts = ts(rs_loss)
ts.plot(rs_loss_ts)
max.lag=round(sqrt(length(rs_loss_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(rs_loss_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(rs_loss_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(rs_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(rs_loss_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(rs_loss_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(rs_loss_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(rs_loss_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)


# selected ARIMA model:
#ARIMA(1,1,1)
fit_ar1_ma1=arima(rs_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)



# BayesOpt with EI Acquisition.....
EI_loss_ts = ts(EI_loss)
ts.plot(EI_loss_ts)
max.lag=round(sqrt(length(EI_loss_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(EI_loss_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(EI_loss_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(EI_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(EI_loss_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(EI_loss_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(EI_loss_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(EI_loss_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
#ARIMA(2,1,0)
fit_ar2=arima(EI_loss_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)


# BayesOpt with MPI Acquisition.....
MPI_loss_ts = ts(MPI_loss)
ts.plot(MPI_loss_ts)
max.lag=round(sqrt(length(MPI_loss_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(MPI_loss_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(MPI_loss_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(MPI_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(MPI_loss_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(MPI_loss_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(MPI_loss_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(MPI_loss_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
#ARIMA(1,1,1)
fit_ar1_ma1=arima(MPI_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

# BayesOpt with LCB Acquisition.....
LCB_loss_ts = ts(LCB_loss)
ts.plot(LCB_loss_ts)
max.lag=round(sqrt(length(LCB_loss_ts))) 
#test for white noise of these 'time series'. We need an ARIMA model:

#ARIMA(1,1,0)
fit_ar1=arima(LCB_loss_ts,order=c(1,1,0),seasonal=c(0,0,0))
Box.test(fit_ar1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1)
BIC(fit_ar1)

#ARIMA(2,1,0)
fit_ar2=arima(LCB_loss_ts,order=c(2,1,0),seasonal=c(0,0,0))
Box.test(fit_ar2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2)
BIC(fit_ar2)

#ARIMA(1,1,1)
fit_ar1_ma1=arima(LCB_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)

#ARIMA(2,1,1)
fit_ar2_ma1=arima(LCB_loss_ts,order=c(2,1,1), seasonal= c(0,0,0))
Box.test(fit_ar2_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma1)
BIC(fit_ar2_ma1)

#ARIMA(2,1,2)
fit_ar2_ma2=arima(LCB_loss_ts,order=c(2,1,2), seasonal= c(0,0,0))
Box.test(fit_ar2_ma2$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma2)
BIC(fit_ar2_ma2)

#ARIMA(2,1,3)
fit_ar2_ma3=arima(LCB_loss_ts,order=c(2,1,3), seasonal= c(0,0,0))
Box.test(fit_ar2_ma3$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma3)
BIC(fit_ar2_ma3)

#ARIMA(2,1,4)
fit_ar2_ma4=arima(LCB_loss_ts,order=c(2,1,4), seasonal= c(0,0,0))
Box.test(fit_ar2_ma4$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar2_ma4)
BIC(fit_ar2_ma4)

#Selected ARIMA model:
#ARIMA(1,1,1)
fit_ar1_ma1=arima(LCB_loss_ts,order=c(1,1,1), seasonal= c(0,0,0))
Box.test(fit_ar1_ma1$residuals,lag=max.lag,type="Ljung-Box")
AIC(fit_ar1_ma1)
BIC(fit_ar1_ma1)






#test for stationarity
library(CADFtest)

CADFtest(rs_loss_ts, type= "drift", criterion= "BIC", max.lag.y=max.lag) # reject null of existence of unit root - stationary

CADFtest(EI_loss, type= "drift", criterion= "BIC", max.lag.y=max.lag) # fail to reject null, existence of unit root

CADFtest(LCB_loss, type= "drift", criterion= "BIC", max.lag.y=max.lag) # reject null of existence of unit root - stationary

CADFtest(MPI_loss, type= "drift", criterion= "BIC", max.lag.y=max.lag) # reject null of existence of unit root - stationary







