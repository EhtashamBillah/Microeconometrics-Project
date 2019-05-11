require(foreign)
require(xlsx)
require(readxl)
require(mice)
require(glmnet)
require(caret)
require(stargazer)
require(car)
require(doSNOW)
require(parallel)
require(dplyr)
require(tidyr)
require(broom)


# setting up parallel computing
getDoParWorkers()
getDoParRegistered()
getDoParName()
numberofcores = detectCores()
cl <- makeCluster(spec = numberofcores, type="SOCK")   # Setting up clusters for Parallel Computing
registerDoSNOW(cl)                                     # Registering clusters
#stopCluster(cl)  



df <- read.delim('NIHMS163142-supplement-01.txt', sep = ",")
colnames(df)

# Converting numeric to categorical
df$pmid <- factor(df$pmid)
df$policy_strength <- factor(df$policy_strength, ordered = T)
df$is_usa_address <- factor(df$is_usa_address)
df$is_nih_funded <- factor(df$is_nih_funded)
df$any_nih_data_sharing_applies <- factor(df$any_nih_data_sharing_applies)
df$any_nih_data_sharing_new_grant <- factor(df$any_nih_data_sharing_new_grant)
df$any_direct_cost_over_500k <- factor(df$any_direct_cost_over_500k)
df$any_new_or_renewed_since_2003 <- factor(df$any_new_or_renewed_since_2003)
df$is_data_shared <- factor(df$is_data_shared)
df$is_usa_address <- factor(df$is_usa_address)
levels(df$is_data_shared) <- c("no", "yes")


##############################################################
# Imputing missing values
##############################################################
imputed_df <- mice(df, m=1, maxit = 200, method = 'cart', seed = 2019)
summary(imputed_df)
df_final <- complete(imputed_df,1)


#### Compute principal components of author experience
# FIRST AUTHOR
df_final$first_career_length = 2008 - df_final$first_first_year
df_final$last_career_length = 2008 - df_final$last_first_year

pc_first = princomp(scale(cbind(log(1+df_final$first_hindex), 
                                log(1+df_final$first_aindex), 
                                df_final$first_career_length)))
summary(pc_first)
first_author_exp = - pc_first$scores[,1]  
pc_first$loadings
#biplot(pc.first)
first_author_exp_freq = ecdf(first_author_exp)(first_author_exp)

# LAST AUTHOR
pc_last = princomp(scale(cbind(log(1+df_final$last_hindex),
                               log(1+df_final$last_aindex), 
                               df_final$last_career_length)))
summary(pc_last)
last_author_exp = - pc_last$scores[,1]
pc_last$loadings
#biplot(pc.last)
last_author_exp_freq = ecdf(last_author_exp)(last_author_exp)

df_final$first_author_exp <- first_author_exp
df_final$last_author_exp <- last_author_exp


#################################################################
# Feature selection using LASSO through cross validation
#################################################################
cv_lasso <- cv.glmnet(x= model.matrix(is_data_shared ~ ., df_final),
                      y = df_final$is_data_shared,
                      family = "binomial",
                      standardize = TRUE,
                      alpha = 1,
                      nfolds = 10,
                      type.measure = "class")

lambda_and_error <- cbind(cv_lasso$lambda,cv_lasso$cvm)
plot(cv_lasso)
coef<-coef(cv_lasso,s='lambda.min',exact=TRUE)
index<-which(coef!=0)
optimum_features <- row.names(coef)[index]    
optimum_features <- optimum_features[-1]
df_lasso <- df_final[,c("impact_factor","policy_strength","first_hindex", "is_data_shared")]

########################################################################
# LOGISTIC REGRESSION
########################################################################
# model used by the author
logistic_model_org <- glm(is_data_shared ~ policy_strength + 
                            is_usa_address * is_nih_funded +
                            any_nih_data_sharing_applies +
                            first_author_exp + last_author_exp+
                            log(1 + sum_of_max_award_for_each_grant) + 
                            log(impact_factor),
                          data = df_final, family = "binomial")

# model with variables selected by LASSO
logistic_model_lasso <- glm(is_data_shared ~ impact_factor + policy_strength +
                              first_hindex, data = df_lasso, family = "binomial")

# The smaller the AIC the better the model
summary(logistic_model_org)
summary(logistic_model_lasso)


# In addition, we can also perform an ANOVA Chi-square 
# test to check the overall effect of variables on the dependent variable.
anova(logistic_model_org, test = 'Chisq')
anova(logistic_model_lasso, test = 'Chisq')

# Compare two models : we can compare both the models using the ANOVA test. 
# Let's say our null hypothesis is that second model is better than the first model. 
anova(logistic_model_org,logistic_model_lasso,test = "Chisq")


######################################################################
# checking the assumptions
######################################################################
#################################################
# A. Original model
probabilities_org <- predict(logistic_model_org,type = "response")
predicted_classes_org <- ifelse(probabilities_org > 0.5, "yes", "no")

# 1. linearity
# selecting only the covariates used in original model
df_org <- df_final[,c("policy_strength","is_usa_address","is_nih_funded","any_nih_data_sharing_applies",
                      "first_author_exp", "last_author_exp","sum_of_max_award_for_each_grant",
                      "impact_factor","is_data_shared")]

# transforming the variables as in original model
df_org$sum_of_max_award_for_each_grant <- log(1 + df$sum_of_max_award_for_each_grant)
df_org$impact_factor <- log(df_org$impact_factor)

# Select only numeric predictors
mydata_org <- df_org %>%
  select_if(is.numeric) 
predictors_org <- colnames(mydata_org)
# Bind the logit and tidying the data for plot
mydata_org <- mydata_org %>%
  mutate(logit = log(probabilities_org/(1-probabilities_org))) %>%
  gather(key = "predictors_org", value = "predictor_values", -logit)

ggplot(mydata_org, aes(logit, predictor_values))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors_org, scales = "free_y")

# 2. MULTICOLLINEARITY
vif(logistic_model_org)

# 3. no extreme values
plot(logistic_model_org, which = 4, id.n = 3)
model_data_org <- augment(logistic_model_org) %>%
  mutate(index = 1:n())
model_data_org %>% top_n(3, .cooksd)

# Plot the standardized residuals:
ggplot(model_data_org, aes(index, .std.resid)) +
  geom_point(aes(color = is_data_shared), alpha = .5) +
  theme_bw()

# Filter potential influential data points (do not exist)
model_data_org%>%
  filter(abs(.std.resid) > 3)
#################################################################

#################################################################
# B. Lasso model
# 1. linearity
probabilities_lasso <- predict(logistic_model_lasso, df_lasso[,-4],type = "response")
predicted_classes_lasso <- ifelse(probabilities_losso > 0.5, "yes", "no")

# Select only numeric predictors
mydata_lasso <- df_lasso %>%
  select_if(is.numeric) 
predictors_lasso <- colnames(mydata_lasso)
# Bind the logit and tidying the data for plot
mydata_lasso <- mydata_lasso %>%
  mutate(logit = log(probabilities_lasso/(1-probabilities_lasso))) %>%
  gather(key = "predictors_lasso", value = "predictor_values", -logit)

ggplot(mydata_lasso, aes(logit, predictor_values))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors_lasso, scales = "free_y")

# 2. MULTICOLLINEARITY
vif(logistic_model_lasso)

# 3. no extreme values
plot(logistic_model_lasso, which = 4, id.n = 3)
model_data_lasso <- augment(logistic_model_lasso) %>%
  mutate(index = 1:n())
model_data_lasso %>% top_n(3, .cooksd)

# Plot the standardized residuals:
ggplot(model_data_lasso, aes(index, .std.resid)) +
  geom_point(aes(color = is_data_shared), alpha = .5) +
  theme_bw()

# Filter potential influential data points (do not exist)
model_data_lasso%>%
  filter(abs(.std.resid) > 3)

