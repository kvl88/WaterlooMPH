library(haven)
library(dplyr)
library(tidyverse)
library(effects)
library(car)
library(lmtest)
library(ggplot2)
library(stargazer)
library(DescTools)

df1 <- read_xpt("LLCP2015.XPT")
df1_sub <- df1[,c("_STATE", "_EDUCAG", "_AGEG5YR", "SEX", "EMTSUPRT", "LSATISFY", "CIMEMLOS", "NUMADULT", "CHILDREN")]
df1_sub <- df1_sub %>% rename(EDUCATION = "_EDUCAG")

df1_sub <- subset(df1_sub,
                  df1_sub$'_STATE' %in% c("27", "44") &
                    df1_sub$'_AGEG5YR' >= 6 & df1_sub$'_AGEG5YR' <= 13)

df1_sub$EDUCATION <- replace(df1_sub$EDUCATION, df1_sub$EDUCATION %in% c(9), NA)
df1_sub$CIMEMLOS <- replace(df1_sub$CIMEMLOS, df1_sub$CIMEMLOS %in% c(7, 9), NA)
df1_sub$CIMEMLOS <- replace(df1_sub$CIMEMLOS, df1_sub$CIMEMLOS %in% c(2), 0)
df1_sub$EMTSUPRT <- replace(df1_sub$EMTSUPRT , df1_sub$EMTSUPRT  %in% c(7, 9), NA)
df1_sub$LSATISFY <- replace(df1_sub$LSATISFY , df1_sub$LSATISFY  %in% c(7, 9), NA)
df1_sub$NUMADULT <- replace(df1_sub$NUMADULT , df1_sub$NUMADULT  >= 2, 2)
df1_sub$CHILDREN <- ifelse(df1_sub$CHILDREN >= 1 & df1_sub$CHILDREN <= 87, 1, df1_sub$CHILDREN)
df1_sub$CHILDREN <- replace(df1_sub$CHILDREN, df1_sub$CHILDREN %in% c(88), 0)
df1_sub$CHILDREN <- replace(df1_sub$CHILDREN, df1_sub$CHILDREN %in% c(99), NA)
df1_final_comb <- na.omit(df1_sub)



df1_final_comb$EDUCATION <- factor(df1_final_comb$EDUCATION)
df1_final_comb$LSATISFY <- factor(df1_final_comb$LSATISFY)
df1_final_comb$EMTSUPRT <- factor(df1_final_comb$EMTSUPRT)
df1_final_comb$SEX<- factor(df1_final_comb$SEX)
df1_final_comb$CIMEMLOS<- factor(df1_final_comb$CIMEMLOS)

#live byself is reference cat
df1_final_comb$NUMADULT<- factor(df1_final_comb$NUMADULT)
#no children is reference cat
df1_final_comb$CHILDREN<- factor(df1_final_comb$CHILDREN)

#BAR CHART BREAKDOWN OF THOSE WITH MEMORY LOSS
memlos_count <- table(df1_final_comb$CIMEMLOS)
memlos_df <- data.frame(CIMEMLOS = factor(names(memlos_count),
                                          labels = c("No SCD", "SCD")),
                        count = as.numeric(memlos_count))
memlos_df$percentage <- memlos_count / sum(memlos_df$count) * 100

ggplot(memlos_df, aes(x = CIMEMLOS, y = count))+
  geom_bar(stat = "identity", fill = "blue") +
  geom_text(aes(label = paste0(count, " (", round(percentage, 2), "%)")), 
            vjust = -0.5, fontface = "bold", color = "red") +
  labs(title = "Subjective Cognitive Decline - Minnesota & Rhode Island 2015 BRFSS",
       x = "Presence",
       y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))


#MAIN GLM MODEL - SCD based on perceived sprt
df1_lm <- glm(CIMEMLOS ~ EMTSUPRT+ EDUCATION +  SEX, data = df1_final_comb, family = "binomial")
df1_lm_reduced <- glm(CIMEMLOS ~ EMTSUPRT, data = df1_final_comb, family = "binomial")
summary(df1_lm)
lr_test <- lrtest(df1_lm, df1_lm_reduced)
print(lr_test)

class(df1_lm)


plot(allEffects(df1_lm))

coefficients <- c(coef(df1_lm))
print(coefficients)

probabilities<- exp(coefficients) / (1 + exp(coefficients))
print(probabilities)

e.out <- allEffects(df1_lm)
e.out$EDUCATION$model.matrix

plogis(e.out$EDUCATION$model.matrix %*% coef(df1_lm))

plot_ed <- predictorEffect("EMTSUPRT", df1_lm)
plot(plot_ed)




#MODEL 2 
df2_lm <- glm(CIMEMLOS ~ EDUCATION + SEX + NUMADULT + CHILDREN, data = df1_final_comb, family = "binomial")
df2_lm_reduced <- glm(CIMEMLOS ~ NUMADULT + CHILDREN, data = df1_final_comb, family = "binomial")
summary(df2_lm)
lr_test2 <- lrtest(df2_lm, df2_lm_reduced)
print(lr_test2)  


coefficients2 <- c(coef(df2_lm))
print(coefficients2)

probabilities2<- exp(coefficients2) / (1 + exp(coefficients2))
print(probabilities2)

plot_ed <- predictorEffect("NUMADULT", df2_lm)
plot(plot_ed)

PseudoR2(df1_lm, which = "CoxSnell")
PseudoR2(df2_lm, which = "CoxSnell")
PseudoR2(df1_lm, which = "McFadden")
PseudoR2(df2_lm, which = "McFadden")