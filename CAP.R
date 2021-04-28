## set working directory
setwd("~/Desktop/Capstone")
getwd()

options("scipen" = 0)
options()$scipen

## loading libraries 
library(randomForest)
require(caTools)
library(caret)
library(C50)
library(pROC)
library(wavelets)
library(ggplot2)
library(animation)
library(readr)
library(tidyr)
library(psych)
library(plyr)
library(dplyr)
library(robustbase)
library(moments)
library(onewaytests)
library(broom)
library(apa)
library(knitr)
library(rmarkdown)
library(reshape2)
# library(dummy_cols)
library(fastDummies)
# library(qcanceRpcR)
library(ggpubr)
library(stringr)
library(tidyverse)
library(psy)
library(igraph)
library(tidytext)
library(topicmodels)
library(tm)
library(syuzhet)
library(wordcloud)
library(SentimentAnalysis)
library(timeordered)
library(widyr)
library(glue)
library(mousetrap)
library(stringi)
library(animation)
library(googleVis)
library(scatterplot3d)
library(corrplot)
library(discretization)
library(DMwR2)

########################################################################################
## Loading the dataset
covid <- read.csv("covid.csv", na.strings = c("null", "NaN", "#N/A", "","None", "NA", " <NA> "))
colnames(covid)
class(covid)    # data.frame
str(covid)
summary(covid)

unique(covid$iso_code)
# 213-11= 202 + 11 Aggregate statistics

counts <- table(covid$continent)
barplot(counts, main="The distribution of continent", xlab="continent")
########################################################################################
## preprocessing

# Transforming data types
# iso_code, continent, location, date -- chr, others are numeric
covid$iso_code <- as.factor(covid$iso_code)
covid$continent <- as.factor(covid$continent)
covid$location <- as.factor(covid$location)
covid$date <- as.Date(covid$date)
str(covid)

# limit the time period (4/1-1/31)
covid <-covid %>% filter(date >= as.Date('2020-04-01') & date <= as.Date('2021-01-31'))

# missing data
mean(is.na(covid)) ## the rate of missing data is 0.4030734
## data preprocess is pretty hard in this dataset because of high missing data rate 

missing.rows <- filter(covid, !complete.cases(covid))
nrow(missing.rows) # All rows contain at least an NA!

sum(apply(covid, 1, function(x) sum(is.na(x)))) 
## the dataset contains 1422956 NA values. 
apply(covid, 2, function(x) sum(is.na(x)))

sort(apply(covid, 2, function(x) sum(is.na(x)))) 
## only three column have not missing value ("iso_code, location, date")

# remove columns (vaccinate and weekly hospitalization data) 
## related vaccination data only contain few time duration and have high missing data rate--> may not be useful for the result --> remove vaccination data
colnames(covid)
covid1 <- covid %>% select(-c(weekly_icu_admissions,weekly_icu_admissions_per_million,weekly_hosp_admissions,weekly_hosp_admissions_per_million,
                               total_vaccinations,people_vaccinated,people_fully_vaccinated,new_vaccinations,new_vaccinations_smoothed,total_vaccinations_per_hundred,
                               people_vaccinated_per_hundred,people_fully_vaccinated_per_hundred,new_vaccinations_smoothed_per_million))


apply(covid1, 2, function(x) sum(is.na(x)))
sort(apply(covid1, 2, function(x) sum(is.na(x))))
mean(is.na(covid1))  
str(covid1)
## after first preprocess, the missing data rate turn to be 0.2394523

# remove columns (smooth_index, gda_per, extreme_poverty, cardiovasc_death_rate,diabetes_prevalence, life_expectancy, tests_per_case,hospital_beds_per_thousand, tests_units ) 
## dependent variables or unrelated to our research
# handwashing_facilities, human_development_index 
covid2 <- covid1 %>% select(-c(new_cases_smoothed,new_deaths_smoothed,new_cases_smoothed_per_million,new_deaths_smoothed_per_million,
                               new_tests_smoothed,new_tests_smoothed_per_thousand,tests_units,gdp_per_capita,extreme_poverty,
                               cardiovasc_death_rate,diabetes_prevalence,hospital_beds_per_thousand, tests_per_case)) 
apply(covid2, 2, function(x) sum(is.na(x)))
sort(apply(covid2, 2, function(x) sum(is.na(x))))
mean(is.na(covid2)) #[1] 0.2575425
summary(covid2)
str(covid2)

# two dataset: the main goal is to manage ICU 
# first dataset: limit the area and countries and select the related variables  (remove dependent variables)
covid3 = covid2 %>% select(iso_code,continent,location,date,total_cases,new_cases,new_cases_per_million, total_deaths,new_deaths,new_deaths_per_million, reproduction_rate,icu_patients,icu_patients_per_million, hosp_patients, hosp_patients_per_million, total_tests,new_tests,new_tests_per_thousand, positive_rate,stringency_index, tests_per_case)
colnames(covid3)

covidicu<- covid3[complete.cases(covid3$icu_patients), ]
covidicu<- covidicu[complete.cases(covidicu$hosp_patients), ]

apply(covidicu, 2, function(x) sum(is.na(x)))
summary(covidicu)
unique(covidicu$iso_code)
sort(table(covidicu$iso_code))
table(covidicu$iso_code)

counts <- table(covidicu$continent)

barplot(counts, main="The distribution of continent",
        xlab="continent")
summary(covidicu)
# Because icu related data did not completely provide in each counties 
# After look through all the data, USA, Canada, UK and some European counties have provided relative completed data (22 counties)
# See the completed data by group (iso_code)

ggplot(covidicu, aes(x=continent, fill=continent)) + geom_bar() + ggtitle("Frequency of the continent")

# BGR GBR IRL AUT BEL CAN CYP CZE EST FRA ITA LUX PRT SVN SWE USA  16/22 (more than 300)
# only one asia country provide icu related data

dev.off() ## bug

# visualization (correlation)
## missing data  --> replace with 0 value!! because of time is more important factor in this dataset
## real time series data (date is important variables in this dataset)

ICU4 <- covidicu[iso_code == c("USA","GBR","CAN","ITA"), ]
ICU5 <- covidicu[iso_code == c("GBR", "CAN", "ITA"), ]

USA <- covidicu %>% filter(iso_code == "USA") 
colnames(USA)
ncol(USA) 
USA1 <- USA[,-c(1:4)]
cor(USA1)
corrplot(cor(USA1),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))

UK <-covidicu %>% filter(iso_code == "GBR") 
head(UK)
UK1 <- UK[,-c(1:4)]
UK1[is.na(UK1)] = 0
cor(UK1)
corrplot(cor(UK1),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))

ITA <- covidicu %>% filter(iso_code == "ITA") 
ITA1 <- ITA[,-c(1:4)]
cor(ITA1)
corrplot(cor(ITA1),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))

Canada <- covidicu %>% filter(iso_code == "CAN") 
Canada1 <- Canada[,-c(1:4)]
Canada1[is.na(Canada1)] = 0
cor(Canada1) ## new tests // total tests --> replace missing data with 0
corrplot(cor(Canada1),method="color", col=colorRampPalette(c("blue","white","red"))(200))
# Check distribution of data with QQ plot

qplot(x=date, y=icu_patients, data=ICU4, geom="line", colour=iso_code, alpha=I(.5), 
      main="Distribution of icu patient by iso_code", xlab="Date", 
      ylab="icu patients")

qplot(x=date, y=new_cases, data=ICU4, geom="line", colour=iso_code, alpha=I(.5), 
      main="Distribution of new cases by iso_code", xlab="Date", 
      ylab="new cases")

qplot(x=date, y=new_cases, data=ICU5, geom="line", colour=iso_code, alpha=I(.5), 
      main="Distribution of new cases by iso_code", xlab="Date", 
      ylab="new cases")

#OCT--> weather --> virus may be easily spread

# one difference about the plot between three countries --> USA have one peak between Jul to Oct 
########## finding reasons !!!!!!!!!
# !!!!!!!!!or possible reasons 

attach(USA)
par(mfrow=c(2,3))
plot(date, icu_patients, main="Scatterplot of icu patients in USA")
plot(date, hosp_patients, main="Scatterplot of hospital patients in USA")
plot(date, new_cases, main="Scatterplot of new cases in USA")
plot(date, new_tests, main="Scatterplot of new tests in USA")
plot(date, total_cases, main="Scatterplot of total cases in USA")
plot(date, total_tests, main="Scatterplot of total tests in USA")

attach(UK)
par(mfrow=c(2,3))
plot(date, icu_patients, main="Scatterplot of icu patients in UK")
plot(date, hosp_patients, main="Scatterplot of hospital patients in UK")
plot(date, new_cases, main="Scatterplot of new cases in UK")
plot(date, new_tests, main="Scatterplot of new tests in UK")
plot(date, total_cases, main="Scatterplot of total cases in UK")
plot(date, total_tests, main="Scatterplot of total tests in UK")

attach(Canada)
par(mfrow=c(2,3))
plot(date, icu_patients, main="Scatterplot of icu patients in Canada")
plot(date, hosp_patients, main="Scatterplot of hospital patients in Canada")
plot(date, new_cases, main="Scatterplot of new cases in Canada")
plot(date, new_tests, main="Scatterplot of new tests in Canada")
plot(date, total_cases, main="Scatterplot of total cases in Canada")
plot(date, total_tests, main="Scatterplot of total tests in Canada")

attach(ITA)
par(mfrow=c(2,3))
plot(date, icu_patients, main="Scatterplot of icu patients in Italy")
plot(date, hosp_patients, main="Scatterplot of hospital patients in Italy")
plot(date, new_cases, main="Scatterplot of new cases in Italy")
plot(date, new_tests, main="Scatterplot of new tests in Italy")
plot(date, total_cases, main="Scatterplot of total cases in Italy")
plot(date, total_tests, main="Scatterplot of total tests in Italy")


# find the day having outlier
Canadaoutlier <-Canada %>% filter(new_cases >= 15000)
nrow(Canadaoutlier)

## Main reason that new cases of Canada and ITA reduce is becasue they decrease new test cases

# UK and USA and ITA have pretty similar result and relationship
# However, in Canada, icu and hosp seems have lower connection with test and cases
# tests and cases are reducing. However, patient still increase as UK, USA, and ITA. 
# reason is because the new cases and new tests has a exime high case
# new cases on 2021-01-03 

ggplot(data=USA,
       aes(x=date, y=value, colour=variable)) +
  geom_line()  

USA %>%
  gather(key,value, new_cases, hosp_patients, icu_patients) %>%
  ggplot(aes(x=date, y=value, colour=key)) +
  geom_line()

UK %>%
  gather(key,value, new_cases, hosp_patients, icu_patients)%>%
  ggplot(aes(x=date, y=value, colour=key)) +
  geom_line()

Canada %>%
  gather(key,value, new_cases, hosp_patients, icu_patients) %>%
  ggplot(aes(x=date, y=value, colour=key)) +
  geom_line()

ITA %>%
  gather(key,value, new_cases, hosp_patients, icu_patients) %>%
  ggplot(aes(x=date, y=value, colour=key)) +
  geom_line()

# 1. a
# icu patient hospital patient new cases --> plot up into same plot 
# therefore, we can estimation the possible incubation period (finding some researches --> 14 days)

# 1. b
# researches to manage patients and predict possible estimation time (6 days)

# 1.c
# OCT -- temperate have related (virus is more active in the winter)

# 1.d
# 

############## second dataset: the correction between variables (covid2 group by country) #############

# 2.1 --> gather countries to verify the relationship between variables 
colnames(covid2)
Countries <- covid2 %>% filter(date == "2021-1-30") 
#summary(Countries)
# remove handwashing_facilities variable, it have more than half value 
# all location and iso_code are different 
str(Countries)
Countriesnew <- Countries[-c(2,10,63,64,88,101, 138, 141, 173, 204),c(2,9,25:28,32,33)]

sort(apply(Countriesnew, 2, function(x) sum(is.na(x))))

# ## missing value
# sort(apply(Countriesnew, 2, function(x) sum(is.na(x))))
# mean(is.na(Countriesnew)) # the dataset contains 0.120603
# 
# Countriesnew1<- Countriesnew[complete.cases(Countriesnew$aged_65_older ), ]
# Countriesnew1<- Countriesnew1[complete.cases(Countriesnew1$human_development_index), ]
# Countriesnew1<- Countriesnew1[complete.cases(Countriesnew1$aged_70_older), ]
# Countriesnew1<- Countriesnew1[complete.cases(Countriesnew1$population_density), ]
# Countriesnew1<- Countriesnew1[complete.cases(Countriesnew1$human_development_index), ]
# 
# sort(apply(Countriesnew1, 2, function(x) sum(is.na(x))))
# nrow(Countriesnew1)

########################################################################
# # outlier
# # histrogram
# par(mfrow=c(1,1))
# hist(Countriesnew1$new_cases_per_million, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$population_density, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$median_age, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$aged_65_older, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$aged_70_older, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$female_smokers, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$male_smokers, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$life_expectancy, main= "distribution of new_cases_per_million")
# hist(Countriesnew1$human_development_index, main= "distribution of new_cases_per_million")
# 
# # 
# boxplot(Countriesnew1$new_cases_per_million, main= "boxplot for new_cases_per_million")
# boxplot(Countriesnew1$population_density, main= "boxplot for new_cases_per_million")
# 
# outliers <- boxplot(Countriesnew1$population_density, plot=FALSE)$out ## out of box plot 
# Countriesnew2 <- Countriesnew1[-which(Countriesnew1$population_density %in% outliers),]
# outliers <- boxplot(Countriesnew2$new_cases_per_million, plot=FALSE)$out
# Countriesnew2 <- Countriesnew2[-which(Countriesnew1$new_cases_per_million %in% outliers),]
# 
# boxplot(Countriesnew2$population_density,main ="Boxplot for Goal")
# boxplot(Countriesnew2$new_cases_per_million, main ="Boxplot for backers_count")
# 
# outliers <- boxplot(Countriesnew2$new_cases_per_million, plot=FALSE)$out
# Countriesnew3 <- Countriesnew2[-which(Countriesnew2$new_cases_per_million %in% outliers),]
# 
# boxplot(Countriesnew3$population_density,main ="Boxplot for Goal")
# boxplot(Countriesnew3$new_cases_per_million, main ="Boxplot for backers_count")
# summary(Countriesnew3)
# 
# Countriesnew3$new_cases_per_million <- ifelse(Countriesnew3$new_cases_per_million > 60.74  , 1, 0)
# 
# ## Plotting the categorial variables after cleaning 
# plot(Countriesnew3$continent)
# 
# str(Countriesnew3)
# ## correlation -- numeric
# Covidcor <- Countriesnew3[, c(3:6, 9,10)]
# write.csv(Covidcor, "Covidcor.csv")
# cor(Covidcor)
# corrplot(cor(Covidcor),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))

########################################################################
CountriesCovid<- Countriesnew[complete.cases(Countriesnew$aged_65_older ), ]
CountriesCovid<- CountriesCovid[complete.cases(CountriesCovid$continent), ]
CountriesCovid<- CountriesCovid[complete.cases(CountriesCovid$aged_70_older), ]
CountriesCovid<- CountriesCovid[complete.cases(CountriesCovid$population_density), ]
CountriesCovid<- CountriesCovid[complete.cases(CountriesCovid$human_development_index), ]

sort(apply(CountriesCovid, 2, function(x) sum(is.na(x))))
nrow(CountriesCovid)
colnames(CountriesCovid)

# outlier
par(mfrow=c(1,1))
boxplot(CountriesCovid$total_cases_per_million, main= "Boxplot for total cases per million each country")
boxplot(CountriesCovid$population_density, main= "Boxplot for population density")

outliers <- boxplot(CountriesCovid$population_density, plot=FALSE)$out ## out of box plot 
CountriesCovid1 <- CountriesCovid[-which(CountriesCovid$population_density %in% outliers),]


boxplot(CountriesCovid1$total_cases_per_million, main ="Boxplot for total cases per million each country")
boxplot(CountriesCovid1$population_density,main ="Boxplot for population density")
summary(CountriesCovid)

CountriesCovid1$total_cases_per_million <- ifelse(CountriesCovid1$total_cases_per_million > 18986.11, 1, 0)
CountriesCovid1$total_cases_per_million <- as.factor(CountriesCovid1$total_cases_per_million)

## Plotting the categorial variables after cleaning 
plot(CountriesCovid1$continent)
colnames(CountriesCovid1)

## correlation -- numeric
Covidcor <- CountriesCovid1[, -c(1,2)]
write.csv(Covidcor, "Covidcor.csv")
cor(Covidcor)
corrplot(cor(Covidcor),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))

dev.off()




################## predictive model: ###################
covid3 = covid2 %>% select(iso_code,continent,location,date,total_cases,new_cases,new_cases_per_million, total_deaths,new_deaths,new_deaths_per_million, reproduction_rate,icu_patients,icu_patients_per_million, hosp_patients, hosp_patients_per_million, total_tests,new_tests,new_tests_per_thousand, positive_rate,stringency_index)
colnames(covid3)

covidicu<- covid3[complete.cases(covid3$icu_patients), ]
covidicu<- covidicu[complete.cases(covidicu$hosp_patients), ]
covidicu<- covidicu[complete.cases(covidicu$positive_rate), ]
covidicu<- covidicu[complete.cases(covidicu$new_tests_per_thousand), ]

apply(covidicu, 2, function(x) sum(is.na(x)))
summary(covidicu)
unique(covidicu$iso_code)
sort(table(covidicu$iso_code))

counts <- table(covidicu$continent)
barplot(counts, main="The distribution of continent",
        xlab="continent")

covidicu1 <- covidicu %>% select(new_cases_per_million, new_deaths_per_million, reproduction_rate,icu_patients_per_million, hosp_patients_per_million, new_tests_per_thousand, positive_rate, stringency_index)
apply(covidicu1, 2, function(x) sum(is.na(x)))
colnames(covidicu1)
cor(covidicu1)
corrplot(cor(covidicu1),method="color",cl.lim=c(-1,1), col=colorRampPalette(c("blue","white","red"))(200))


################## predictive model: ###################

# split data for training and testing set
size <- floor(0.75*nrow(covidicu1))
training_index <- sample(nrow(covidicu1), size = size, replace = FALSE)
train <- covidicu1[training_index,]
test <- covidicu1[-training_index,]

########################################################
# Linear regression
icureg <- lm(icu_patients_per_million ~ . , data = train)
summary(icureg)

cor(predict(icureg, newdata=test),test$icu_patients_per_million)^2
dev.off()

pr.lm <- predict(icureg,test)
plot(test$icu_patients_per_million,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

MSE.lm <- sum((pr.lm - test$icu_patients_per_million)^2)/nrow(test)
MSE.lm
MAE(pr.lm, test$icu_patients_per_million)

##  decision trees 
# install.packages("rsample")
# install.packages("earth")
# install.packages("caret")
# install.packages("vip")
# install.packages("pdp")

library(rsample)   # data splitting 
library(ggplot2)   # plotting
library(earth)     # fit MARS models
library(caret)     # automating the tuning process
library(vip)       # variable importance
library(pdp)       # variable relationships

##  regression trees 
library(rpart)

# creating the model
dt <- rpart(icu_patients_per_million~., method="anova", data=train)
printcp(dt) # display the results

plotcp(dt) # visualize cross-validation results

par(mfrow=c(1,1)) 
ㄎrsq.rpart(dt) # visualize cross-validation results  

summary(dt)
# plot tree
plot(dt, uniform=TRUE,
     main="Regression Tree for icu_patient ratio")
text(dt, use.n=TRUE, all=TRUE, cex=.6)

cor(predict(dt, newdata=test),test$icu_patients_per_million)^2

pr.dt<- predict(dt,test)
MSE.dt <- sum((pr.dt - test$icu_patients_per_million)^2)/nrow(test)
MAE(pr.dt, test$icu_patients_per_million)


##  random forest 
rf <- randomForest(icu_patients_per_million ~ . , data = train)
rf

cor(predict(rf, newdata=test),test$icu_patients_per_million)^2

pr.rf <- predict(rf,test)
MSE.rf <- sum((pr.rf - test$icu_patients_per_million)^2)/nrow(test)
MSE.rf 
MAE(pr.rf, test$icu_patients_per_million)
## regression artifical neural network
library(tidyverse)
library(neuralnet)
library(GGally)

# Normalize the data
maxs <- apply(covidicu1, 2, max) 
mins <- apply(covidicu1, 2, min)
scaled <- as.data.frame(scale(covidicu1, center = mins, scale = maxs - mins))

set.seed(1201)
index <- sample(1:nrow(covidicu1), round(0.75 * nrow(covidicu1)))
train <- scaled[index,]
test <- scaled[-index,]
colnames(test)

# Build Neural Network
nn <- neuralnet(icu_patients_per_million ~ new_cases_per_million + new_deaths_per_million + reproduction_rate + hosp_patients_per_million + new_tests_per_thousand + positive_rate + stringency_index, data = train, hidden = c(5, 3), linear.output = TRUE)  

# Plot the neural network
plot(nn)

# Compute mean squared error/ mean absolute error
pr.nn <- compute(nn, test[,-4])
pr.nn_ <- pr.nn$net.result * (max(covidicu1$icu_patients_per_million) - min(covidicu1$icu_patients_per_million)) + min(covidicu1$icu_patients_per_million)
test.r <- (test$icu_patients_per_million) * (max(covidicu1$icu_patients_per_million) - min(covidicu1$icu_patients_per_million)) + 
  min(covidicu1$icu_patients_per_million)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test)
MSE.nn
MAE(test.r, pr.nn_)
R2(test.r, pr.nn_, form = "traditional")


# Plot regression line
plot(test$icu_patients_per_million, pr.nn_, col = "red", 
     main = 'Real vs Predicted')
abline(0,110)  
