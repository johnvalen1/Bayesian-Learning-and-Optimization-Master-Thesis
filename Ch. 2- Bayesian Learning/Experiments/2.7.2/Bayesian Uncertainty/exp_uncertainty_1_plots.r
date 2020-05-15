#This creates plots for Experiment 2.7.1, showing the two forms of uncertainty for a Bayesian neural
#network before and after being endowed with the ability to refuse making predictions on examples it may
#consider more 'exotic'.
library(ggplot2)
#Before: note that all we control here are the number of samples from the variational posterior at test-time
samples_taken = seq(10,30,5)
avg_epistemic = c(0.00201,
                  0.00380,
                  0.00075,
                  0.00095,
                  0.00176)
avg_aleatoric = c(0.00464,
                  0.00652,
                  0.00292,
                  0.00375,
                  0.00377)

df = data.frame(
  samples_taken = samples_taken,
  avg_epistemic = avg_epistemic,
  avg_aleatoric = avg_aleatoric)

ggplot(df, aes(samples_taken)) + 
  geom_line(aes(y = avg_epistemic, colour = "Average epistemic uncertainty")) + 
  geom_line(aes(y = avg_aleatoric, colour = "Average aleatoric uncertainty")) + 
  labs(title="Uncertainties", x ="Number of samples from var. posterior", y = "Avg. Uncertainties") + theme(plot.title = element_text(hjust = 0.5))



#After: not much of a relationship was seen between #samples and uncertainties, so we implement a threshold 
#that governs how picky the network is. Higher threshold, more picky.
threshold = seq(0.1, 0.6, 0.1)
avg_epistemic = c(0.00110,
                  0.00142,
                  0.00209,
                  0.00095,
                  0.00057,
                  0.00097)
avg_aleatoric = c(0.00264,
                  0.00466,
                  0.00536,
                  0.00287,
                  0.00221,
                  0.00336)

df = data.frame(
  threshold = threshold,
  avg_epistemic = avg_epistemic,
  avg_aleatoric = avg_aleatoric)

ggplot(df, aes(threshold)) + 
  geom_line(aes(y = avg_epistemic, colour = "Average epistemic uncertainty")) + 
  geom_line(aes(y = avg_aleatoric, colour = "Average aleatoric uncertainty")) + 
  labs(title="Uncertainties when network can refuse", x ="Threshold", y = "Avg. Uncertainties") + theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = threshold[seq(1, length(threshold), 1)]) 
  






