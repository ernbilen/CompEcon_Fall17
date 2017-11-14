# Clear workspace
rm(list=ls())

# Install necessary packages if not already installed
#install.packages("pwt9")
#install.packages("ggplot2")
#install.packages("grid")
#install.packages("gridExtra")
library("texreg")
library("lmtest")
library("grid")
library("gridExtra")
library("pwt9")

# Get Penn World Table 9.0 as a dataframe
pwt <- pwt9.0

# Countries of interest
nonoil = c("Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", 
           "Central African Republic", "Congo, Democratic Republic", "Egypt", "Ethiopia", 
           "Ghana", "Cote d'Ivoire", "Kenya", "Liberia", "Madagascar", "Malawi", "Mali", 
           "Mauritania", "Mauritius", "Morocco", "Mozambique", "Niger", "Nigeria", "Rwanda", 
           "Senegal", "Sierra Leone", "South Africa", "Sudan (Former)", "U.R. of Tanzania: Mainland", 
           "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe", "Bangladesh", 
           "China, Hong Kong SAR", "India", "Israel", "Japan", "Jordan", 
           "Republic of Korea", "Malaysia", "Nepal", "Pakistan", 
           "Philippines", "Singapore", "Sri Lanka", "Syrian Arab Republic", "Thailand", "Austria", 
           "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Ireland", "Italy", 
           "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland", "Turkey", 
           "United Kingdom", "Canada", "Costa Rica", "Dominican Republic", "El Salvador", 
           "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", 
           "Trinidad and Tobago", "United States of America", "Argentina", "Bolivia (Plurinational State of)", 
           "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", 
           "Venezuela (Bolivarian Republic of)", "Australia", "Indonesia", "New Zealand")

inter = c("Algeria", "Botswana", "Cameroon", "Ethiopia", 
          "Cote d'Ivoire", "Kenya", "Madagascar", "Malawi", "Mali", "Morocco", "Nigeria", 
          "Senegal",  "South Africa", "U.R. of Tanzania: Mainland", 
          "Tunisia", "Zambia", "Zimbabwe", "Bangladesh", 
          "China, Hong Kong SAR", "India", "Israel", "Japan", "Jordan", 
          "Republic of Korea", "Malaysia", "Pakistan", 
          "Philippines", "Singapore", "Sri Lanka", "Syrian Arab Republic", "Thailand", "Austria", 
          "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Ireland", "Italy", 
          "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland", "Turkey", 
          "United Kingdom", "Canada", "Costa Rica", "Dominican Republic", "El Salvador", 
          "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", 
          "Trinidad and Tobago", "United States of America", "Argentina", "Bolivia (Plurinational State of)", 
          "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", 
          "Venezuela (Bolivarian Republic of)", "Australia", "Indonesia", "New Zealand")

oecd <- c("Australia","Austria","Belgium","Canada","Denmark","Finland","France", "Germany",
          "Greece","Ireland","Italy","Japan","Netherlands","New Zealand","Norway","Portugal",
          "Spain","Sweden","Switzerland","Turkey","United Kingdom","United States of America")



# Restricting years to 1960-2014
pwt = pwt[(pwt$year>=1960)&(pwt$year<=2014),]

# Restricting sample to nonoil countries
pwt <- subset(pwt, pwt[[1]] %in% nonoil, drop = TRUE)

# Sorting data alphabetically by country name
pwt=as.matrix(pwt)
pwt <- pwt[order(pwt[,1]),]
pwt=data.frame(pwt)

# Converting variables of interest from factor to numeric
pwt$investment <- as.numeric(as.character(pwt$csh_i))
pwt$investment_hc <- as.numeric(as.character(pwt$hc))
pwt$income <- as.numeric(as.character(pwt$rgdpna))

# Calculating average savings through investment as in MRW
s=aggregate(investment ~ country, pwt, mean)
s = s[,2] *100
X.1 <- log(s)

# Calculating average human capital investment
h=aggregate(investment_hc ~ country, pwt, mean)
h=h[,2]
X.3 <- log(h)

# Real GDP in 2014
Y=pwt[pwt$year==2014,]$income

# Converting Real GDP to original values as they are reported in millions in PWT
Y=Y*1000000

# Obtaining working age population data from the World Bank's World Development Indicators
# Install necessary WDI package if not already installed
#install.packages("WDI")
library("WDI")

# Get variables of interest
work <- WDI(country = "all", indicator = c("SP.POP.1564.TO", "SP.POP.1564.TO.ZS"),
start = 1960, end = 2014, extra = FALSE, cache = NULL)

# Choosing countries of interest with WDI names
nonoil2 = c("Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", 
            "Central African Republic", "Congo, Dem. Rep.", "Egypt, Arab Rep.", "Ethiopia", 
            "Ghana", "Cote d'Ivoire", "Kenya", "Liberia", "Madagascar", "Malawi", "Mali", 
            "Mauritania", "Mauritius", "Morocco", "Mozambique", "Niger", "Nigeria", "Rwanda", 
            "Senegal", "Sierra Leone", "South Africa", "Sudan", "Tanzania", 
            "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe", "Bangladesh", 
            "Hong Kong SAR, China", "India", "Israel", "Japan", "Jordan", 
            "Korea, Rep.", "Malaysia", "Nepal", "Pakistan", 
            "Philippines", "Singapore", "Sri Lanka", "Syrian Arab Republic", "Thailand", "Austria", 
            "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Ireland", "Italy", 
            "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland", "Turkey", 
            "United Kingdom", "Canada", "Costa Rica", "Dominican Republic", "El Salvador", 
            "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", 
            "Trinidad and Tobago", "United States", "Argentina", "Bolivia", "Brazil", 
            "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", 
            "Venezuela, RB", "Australia", "Indonesia", "New Zealand")

inter2 = c("Algeria", "Botswana", "Cameroon", "Ethiopia", 
           "Cote d'Ivoire", "Kenya", "Madagascar", "Malawi", "Mali", "Morocco", "Nigeria", 
           "Senegal", "South Africa", "U.R. of Tanzania: Mainland", 
           "Tunisia", "Zambia", "Zimbabwe", "Bangladesh", 
           "China, Hong Kong SAR", "India", "Israel", "Japan", "Jordan", 
           "Republic of Korea", "Malaysia",  "Pakistan", 
           "Philippines", "Singapore", "Sri Lanka", "Syrian Arab Republic", "Thailand", "Austria", 
           "Belgium", "Denmark", "Finland", "France", "Germany", "Greece", "Ireland", "Italy", 
           "Netherlands", "Norway", "Portugal", "Spain", "Sweden", "Switzerland", "Turkey", 
           "United Kingdom", "Canada", "Costa Rica", "Dominican Republic", "El Salvador", 
           "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", 
           "Trinidad and Tobago", "United States", "Argentina", "Bolivia", "Brazil", 
           "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", 
           "Venezuela (Bolivarian Republic of)", "Australia", "Indonesia", "New Zealand")

oecd2 <- c("Australia","Austria","Belgium","Canada","Denmark","Finland","France", "Germany",
           "Greece","Ireland","Italy","Japan","Netherlands","New Zealand","Norway","Portugal",
           "Spain","Sweden","Switzerland","Turkey","United Kingdom","United States")


# Restricting sample to nonoil countries
work <- subset(work, work[[2]] %in% nonoil2, drop = TRUE)

# Restricting years to 1960-2014
work = work[(work$year>="1960")&(work$year<="2014"),]

# Rename WDI variables as in PWT to merge them accordingly
work$country[work$country=='Hong Kong SAR, China'] <- 'China, Hong Kong SAR'
work$country[work$country=='Korea, Rep.'] <- 'Republic of Korea'
work$country[work$country=='Congo, Dem. Rep.'] <- 'Congo, Democratic Republic'
work$country[work$country=='Venezuela, RB'] <- 'Venezuela (Bolivarian Republic of)'
work$country[work$country=='Egypt, Arab Rep.'] <- 'Egypt'
work$country[work$country=='Sudan'] <- 'Sudan (Former)'
work$country[work$country=='Tanzania'] <- 'U.R. of Tanzania: Mainland'


# Sorting alphabetically by country
work=as.matrix(work)
work <- work[order(work[,2]),]
work=data.frame(work)

# Converting variables of interest from factor to numeric
work$work_pop <- as.numeric(as.character(work$SP.POP.1564.TO))
work$work_pop_frac <- as.numeric(as.character(work$SP.POP.1564.TO.ZS))

# Creating dataframes for the years of interest
work1960 = work[work$year==1960,]
work2014 = work[work$year==2014,]

# Working age population in 2014
L = work2014$work_pop

# Output per worker y=Y/L in 2014
y = Y/L
y <- as.matrix(y,nrow=length(nonoil),ncol=1)

# Population growth n
work$pct_change_n <- ave(work$work_pop, work$country, FUN=function(x) c( NA, diff(x)/x[-length(x)])  )
n=aggregate(pct_change_n ~ country, work, mean)
n=n[,2]

# Creating n+g+d where g+d=.05 as in MRW
n_g_d = n + .05
X.2 = log(n_g_d)

# Starting y
Y0 = pwt[pwt$year==1960,]$income
Y0=Y0*1000000
L0 = work1960$work_pop
y0 = Y0/L0

# Average growth in y
G = log(y) - log(y0)

# Running the first set of models
# Baseline model (Textbook Solow Model)
reg1a = lm(log(y)~X.1+X.2)
# Applying restriction as in MRW
reg1a_r = lm(log(y)~(X.1-X.2))
# Augmented model
reg1b = lm(log(y)~X.1+X.3+X.2)
# Applying restriction
reg1b_r = lm(log(y)~(X.1-X.2) + (X.3-X.2))
# Unconditional growth
reg1c = lm(G~log(y0))
# Conditional growth
reg1d = lm(G~log(y0)+X.1+X.2)
# Conditional growth with human capital investment
reg1e = lm(G~log(y0)+X.1+X.3+X.2)

# Implied alphas
alpha1a = (reg1a_r$coefficients[2])/(1+reg1a_r$coefficients[2])
alpha1a = format(round(alpha1a, 2), nsmall = 2)
m1 = reg1b_r$coefficients[2]
m2 = reg1b_r$coefficients[3]
alpha1b = (m1*(1-m2/(1+m2))) / (1+m1*(1-m2/(1+m2)))
beta1b = (1-alpha1b) * m2 / (1+m2)
alpha1b = format(round(alpha1b, 2), nsmall = 2)
beta1b = format(round(beta1b, 2), nsmall = 2)

# Restricting sample to "intermediate" countries
pwt <- subset(pwt, pwt[[1]] %in% inter, drop = TRUE)
work <- subset(work, work[[2]] %in% inter2, drop = TRUE)

# Recalculating variables
rm(s,X.1,L,n,X.2,Y,y,n_g_d,h,Y0,L0,y0,G,m1,m2)
s=aggregate(investment ~ country, pwt, mean)
s = s[,2] *100
X.1 <- log(s)

# Average human capital investment
h=aggregate(investment_hc ~ country, pwt, mean)
h=h[,2]
X.3 <- log(h)

# Real GDP in 2014
Y=pwt[pwt$year==2014,]$income
Y=Y*1000000

# Working age population in 2014
work2014 = work[work$year==2014,]
L = work2014$work_pop

# Output per worker y=Y/L in 2014
y = Y/L
y <- as.matrix(y,nrow=length(inter),ncol=1)

# Population growth n
work$pct_change_n <- ave(work$work_pop, work$country, FUN=function(x) c( NA, diff(x)/x[-length(x)])  )
n=aggregate(pct_change_n ~ country, work, mean)
n=n[,2]
n_g_d = n + .05
X.2 = log(n_g_d)

# Starting y
Y0 = pwt[pwt$year==1960,]$income
Y0=Y0*1000000
work1960 = work[work$year==1960,]
L0 = work1960$work_pop
y0 = Y0/L0

# Average growth in y
G = log(y) - log(y0)

# Running the second set of models
reg2a = lm(log(y)~X.1+X.2)
reg2a_r = lm(log(y)~(X.1-X.2))
reg2b = lm(log(y)~X.1+X.3+X.2)
reg2b_r = lm(log(y)~(X.1-X.2) + (X.3-X.2))
reg2c = lm(G~log(y0))
reg2d = lm(G~log(y0)+X.1+X.2)
reg2e = lm(G~log(y0)+X.1+X.3+X.2)

# Implied alphas
alpha2a = (reg2a_r$coefficients[2])/(1+reg2a_r$coefficients[2])
alpha2a = format(round(alpha2a, 2), nsmall = 2)
m1 = reg2b_r$coefficients[2]
m2 = reg2b_r$coefficients[3]
alpha2b = (m1*(1-m2/(1+m2))) / (1+m1*(1-m2/(1+m2)))
beta2b = (1-alpha2b) * m2 / (1+m2)
alpha2b = format(round(alpha2b, 2), nsmall = 2)
beta2b = format(round(beta2b, 2), nsmall = 2)

# Plotting log savings and log real income for intermediate countries as in MRW
# Denoting Y in trillions
Y = Y / 1000000000000
library(ggplot2)
plot0=ggplot(,aes(x = X.1, y = log(y))) + geom_point(aes(size = Y)) + xlab('ln(s)') + ylab('ln(y)') + 
  xlim(c(2,4)) + ylim(c(7,12))

# Plotting unconditional convergence
plot1=ggplot(,aes(x = log(y0), y = G)) + geom_point() + 
  xlab('Log output per working age adult: 1960') + ylab('Growth rate: 1960-2014')  +
  xlim(c(6,11)) + ylim(c(-1,3))+
  labs(title="Unconditional Convergence")

# Plotting conditional convergence
plot2=ggplot(,aes(x = log(y0) , y = predict.lm(reg2d))) + geom_point() + 
  xlab('Log output per working age adult: 1960') + ylab('Growth rate: 1960-2014')  +
  xlim(c(6,11)) + ylim(c(-1,3)) +
  labs(title="Conditional Convergence")

# Plotting conditional convergence with human capital
plot3=ggplot(,aes(x = log(y0) , y = predict.lm(reg2e))) + geom_point() + 
  xlab('Log output per working age adult: 1960') + ylab('Growth rate: 1960-2014')  +
  xlim(c(6,11)) + ylim(c(-1,3)) +
  labs(title="Conditional Convergence w/Human Capital")

# Plotting convergence plots together
grid.arrange(plot1,plot2,plot3, ncol=1)

# Restricting sample to oecd countries
pwt <- subset(pwt, pwt[[1]] %in% oecd, drop = TRUE)
work <- subset(work, work[[2]] %in% oecd2, drop = TRUE)

# Recalculating the variables
rm(s,X.1,L,n,X.2,Y,y,n_g_d,h,Y0,L0,y0,G,m1,m2)
s=aggregate(investment ~ country, pwt, mean)
s = s[,2] *100
X.1 <- log(s)

# Average human capital investment
h=aggregate(investment_hc ~ country, pwt, mean)
h=h[,2]
X.3 <- log(h)

# Real GDP in 2014
Y=pwt[pwt$year==2014,]$income
Y=Y*1000000

# Working age population in 2014
work2014 = work[work$year==2014,]
L = work2014$work_pop

# Output per worker y=Y/L
y = Y/L
y <- as.matrix(y,nrow=length(oecd),ncol=1)

# Population growth n
work$pct_change_n <- ave(work$work_pop, work$country, FUN=function(x) c( NA, diff(x)/x[-length(x)])  )
n=aggregate(pct_change_n ~ country, work, mean)
n=n[,2]
n_g_d = n + .05
X.2 = log(n_g_d)

# Starting y
Y0 = pwt[pwt$year==1960,]$income
Y0=Y0*1000000
work1960 = work[work$year==1960,]
L0 = work1960$work_pop
y0 = Y0/L0

# Average growth in y
G = log(y) - log(y0)

# Running the third set of models
reg3a = lm(log(y)~X.1+X.2)
reg3a_r = lm(log(y)~(X.1-X.2))
reg3b = lm(log(y)~X.1+X.3+X.2)
reg3b_r = lm(log(y)~(X.1-X.2) + (X.3-X.2))
reg3c = lm(G~log(y0))
reg3d = lm(G~log(y0)+X.1+X.2)
reg3e = lm(G~log(y0)+X.1+X.3+X.2)

# Implied alphas
alpha3a = (reg3a_r$coefficients[2])/(1+reg3a_r$coefficients[2])
alpha3a = format(round(alpha3a, 2), nsmall = 2)
m1 = reg3b_r$coefficients[2]
m2 = reg3b_r$coefficients[3]
alpha3b = (m1*(1-m2/(1+m2))) / (1+m1*(1-m2/(1+m2)))
beta3b = (1-alpha3b) * m2 / (1+m2)
alpha3b = format(round(alpha3b, 2), nsmall = 2)
beta3b = format(round(beta3b, 2), nsmall = 2)


# Overall results
# Baseline model
print(texreg(list(reg1a, reg2a, reg3a), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:1", caption = "Baseline results.",
             custom.note = paste(alpha1a,alpha2a,alpha3a), float.pos = "hb!"))

# Baseline model(restricted)
print(texreg(list(reg1a_r, reg2a_r, reg3a_r), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:2", caption = "Baseline results.",
             custom.note = paste(alpha1a,alpha2a,alpha3a), float.pos = "hb!"))

# Augmented model
print(texreg(list(reg1b, reg2b, reg3b), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:3", caption = "Augmented model results.",
             custom.note = paste(alpha1b,alpha2b,alpha3b,beta1b,beta2b,beta3b), float.pos = "hb!"))

# Augmented model(restricted)
print(texreg(list(reg1b_r, reg2b_r, reg3b_r), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:4", caption = "Augmented model results.",
             custom.note = paste(alpha1b,alpha2b,alpha3b,beta1b,beta2b,beta3b), float.pos = "hb!"))


# Growth 1: Unconditional
print(texreg(list(reg1c, reg2c, reg3c), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:5", caption = "Unconditional convergence results.",
             float.pos = "hb!"))

# Growth 2: Conditional 
print(texreg(list(reg1d, reg2d, reg3d), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:6", caption = "Conditional convergence results.",
             float.pos = "hb!"))

# Growth 2: Conditional with human capital
print(texreg(list(reg1e, reg2e, reg3e), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:7", caption = "Conditional convergence with hc results.",
             float.pos = "hb!"))
