# Clear the workspace
rm(list=ls())

# Install the necessary packages, if not already installed
# install.packages("mfx")
# install.packages("texreg")

library(mfx)
library(texreg)

# Reading in data
dataset <- read.csv(file="c:/users/master/desktop/ps6/tdhs.csv", header=TRUE, sep=",",stringsAsFactors = FALSE)

# Dropping high-school students
dataset = dataset[dataset$age>=17,]

# Generating headscarf dummy
dataset$headscarf = as.numeric(dataset$headscarf=="Regular")

# Generating a treatment dummy
dataset$dummy = NA
dataset$dummy = as.numeric(dataset$birth>=1990)

# Generating  the interaction term necessary for diff-in-diff
dataset$dummyheadscarf = NA
dataset$dummyheadscarf = dataset$dummy * dataset$headscarf

# Creating conservatism-proxy dummy
dataset$dayak_kavga = as.numeric(dataset$dayak_kavga=="Yes")

# Model 1: Base model
r1 <- glm(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married 
          + anneuni + babauni, family = binomial(link = "probit"), data = dataset)

# Model 1 ME
r1_me <- probitmfx(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married
                   + anneuni + babauni, data = dataset)

# Model 2: Controlling for conservatism
r2 <- glm(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married 
          + anneuni + babauni + dayak_kavga, family = binomial(link = "probit"), data = dataset)

# Model 2 ME
r2_me <- probitmfx(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married
                   + anneuni + babauni + dayak_kavga, data = dataset)

# Model 3: Closer cohorts
dataset = dataset[dataset$birth>=1984,]

r3 <- glm(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married 
          + anneuni + babauni + dayak_kavga, family = binomial(link = "probit"), data = dataset)

# Model 3 ME
r3_me <- probitmfx(collegegrad ~ headscarf + dummy + dummyheadscarf + age + married
                   + anneuni + babauni + dayak_kavga, data = dataset)

# Export results to LaTeX
print(texreg(list(r1_me, r2_me, r3_me), dcolumn = TRUE, booktabs = TRUE,
             use.packages = FALSE, label = "tab:1", caption = "Regression results.",
             float.pos = "hb!"))





