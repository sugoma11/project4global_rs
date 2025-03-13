dd <- read.csv("dd_ndvi.csv")$x
eg <- read.csv("eg_ndvi.csv")$x


par(mfrow = c(1,2))

hist(dd, main = "NDVI for deciduous vegetation", col = "green", border="black", xlab = "NDVI", breaks=32, xlim=c(0.5,1))
abline(v = mean(dd), col = "black", lwd = 2, lty = 2)

text(mean(dd-0.2), max(hist(dd, plot = FALSE, breaks=32)$counts) * 0.9, 
     labels = paste("Mean =", round(mean(dd), 2)), 
     pos = 4, col = "black")


hist(eg, main = "NDVI for evergreen vegetation", col = "darkgreen", border="black", xlab = "NDVI", breaks=32, xlim=c(0.5,1))
abline(v = mean(eg), col = "black", lwd = 2, lty = 2)

text(mean(eg), max(hist(eg, plot = FALSE, breaks=32)$counts) * 0.9, 
     labels = paste("Mean =", round(mean(eg), 2)), 
     pos = 2, col = "black")

