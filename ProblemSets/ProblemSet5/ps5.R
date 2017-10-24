# Reading in the data
dataset <- read.csv(file="c:/users/eren/desktop/tdhs.csv", header=TRUE, sep=",")

# Plot 1
# Plotting college attainment rate for each group on the same graph
plot(dataset[,1],dataset[,2],type="o",col='blue',xlim=c(1959,1994),ylim=c(0,1),xaxt='n',yaxt='n',xlab='',ylab='')
par(new=t)
plot(dataset[,1],dataset[,3],type="o",col='red',xlim=c(1959,1994),ylim=c(0,1),xaxt='n',yaxt='n',xlab='',ylab='')
par(new=t)
plot(dataset[,1],dataset[,4],type="o",col='green',xlim=c(1959,1994),ylim=c(0,1),xaxt='n',yaxt='n',xlab='cohorts',ylab='college attainment rate')

# Putting the legends box
legend('topright', legend=c('without headscarf','with headscarf','male'),cex=.75, col=c('red','blue','green'),lty=c(1,1,1))

# Generating the axes
text(x=seq(1959, 1994, by=1), par("usr")[3] - 0.02,labels = seq(1959,1994,1) , srt = 45, pos = 1, xpd = TRUE)
axis(side=2, at=seq(0,1,.05))

# Creating vertical lines to show the period in which the ban was in effect
abline(v=1980,col="red",lty=3)
abline(v=1990,col="red",lty=3)

# Plot 2
# Plotting headscarf wearing rate among college graduates vs. non-college graduates
plot(dataset[,1],dataset[,5],type="o",col='red',xlim=c(1959,1994),ylim=c(0,1),xaxt='n',yaxt='n',xlab='',ylab='')
par(new=t)
plot(dataset[,1],dataset[,6],type="o",col='blue',xlim=c(1959,1994),ylim=c(0,1),xaxt='n',yaxt='n',xlab='cohorts',ylab='headscarf rate')

# Putting the legends box
legend('topright', legend=c('no college educ','college educ'),cex=.75, col=c('red','blue'),lty=c(1,1))

# Generating the axes
text(x=seq(1959, 1994, by=1), par("usr")[3] - 0.02,labels = seq(1959,1994,1) , srt = 45, pos = 1, xpd = TRUE)
axis(side=2, at=seq(0,1,.05))

# Creating vertical lines to show the period in which the ban was in effect
abline(v=1980,col="red",lty=3)
abline(v=1990,col="red",lty=3)

# Plot 3
# Plotting labor force participation rate over time
plot(dataset[,1],dataset[,7],type="o",col='red',xlim=c(1985,2013),ylim=c(0,.6),xaxt='n',yaxt='n',xlab='',ylab='')
par(new=t)
plot(dataset[,1],dataset[,8],type="o",col='blue',xlim=c(1985,2013),ylim=c(0,.6),xaxt='n',yaxt='n',xlab='year',ylab='LFPR')

# Putting the legends box
legend('topright', legend=c('without headscarf','with headscarf'),cex=.75, col=c('blue','red'),lty=c(1,1))

# Generating the axes
text(x=seq(1985, 2013, by=1), par("usr")[3] - 0.02,labels = seq(1985,2013,1) , srt = 45, pos = 1, xpd = TRUE)
axis(side=2, at=seq(0,.6,.05))

