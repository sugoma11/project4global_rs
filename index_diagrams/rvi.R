dd <- read.csv("dd_rvi.csv")$x
eg <- read.csv("eg_rvi.csv")$x


par(mfrow = c(1,2))

hist(dd, main = "RVI for deciduous vegetation", col = "green", border="black", xlab = "RVI", breaks=32, xlim=c(0.4,0.8))
abline(v = mean(dd), col = "black", lwd = 2, lty = 2)

text(mean(dd-0.2), max(hist(dd, plot = FALSE, breaks=32)$counts) * 0.9, 
     labels = paste("Mean =", round(mean(dd), 2)), 
     pos = 4, col = "black")


hist(eg, main = "RVI for evergreen vegetation", col = "darkgreen", border="black", xlab = "RVI", breaks=32, xlim=c(0.4,0.8))
abline(v = mean(eg), col = "black", lwd = 2, lty = 2)

text(mean(eg), max(hist(eg, plot = FALSE, breaks=32)$counts) * 0.9, 
     labels = paste("Mean =", round(mean(eg), 2)), 
     pos = 2, col = "black")

