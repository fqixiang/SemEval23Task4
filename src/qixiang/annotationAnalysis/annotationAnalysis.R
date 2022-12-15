library(tidyverse)
library(readxl)
library(irr)

#import the annotations and merge them
aRandomSampleByQ <- read_excel("aRandomSampleQixiang.xlsx") %>% 
  select(-Comment) %>% 
  mutate(coder = "Q")

lowPerformanceSampleByQ <- read_excel("randomSamplesLowPerformanceQixiang.xlsx") %>% 
  select(-Comments) %>% 
  mutate(coder = "Q")

aRandomSampleByC <- read_excel("aRandomSampleChristian.xlsx") %>% 
  mutate(coder = "C")

lowPerformanceSampleByC <- read_excel("randomSamplesLowPerformanceChristian.xlsx") %>% 
  mutate(coder = "C")


# compute simple agreement
computeSimpleAgreement <- function(value, dataset1, dataset2, message = TRUE) {
  labels1 <- pull(dataset1, value)
  labels2 <- pull(dataset2, value)
  
  if (message) {
    message <- paste(value, "->", round(mean(labels1 == labels2), 2))
    return(message)
  }
  
  return(mean(labels1 == labels2))
}

# krippendorf's alpha
computeKrippendorffsAlpha <- function(value, dataset1, dataset2, message = TRUE) {
  dataset <- bind_rows(dataset1, dataset2) %>% 
    select('Argument ID', value, coder) %>% 
    pivot_wider(id_cols = coder, names_from = 'Argument ID', values_from = value) %>% 
    select(-coder) %>% 
    as.matrix()
  
  if (message) {
    message <- paste(value, "->", round(kripp.alpha(dataset, method = "nominal")$value, 2))
    return(message)
  }
  
  return(kripp.alpha(dataset, method = "nominal")$value)
}

#compare between Q and C
valueNames <- c("Self-direction: thought", "Self-direction: action",
                "Stimulation", "Hedonism", "Achievement",
                "Power: dominance", "Power: resources", "Face",
                "Security: personal", "Security: societal", "Tradition",
                "Conformity: rules", "Conformity: interpersonal", "Humility",
                "Benevolence: caring", "Benevolence: dependability", "Universalism: concern",
                "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity" )

lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByQ, aRandomSampleByC) %>% 
  unlist()

lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByQ, aRandomSampleByC, FALSE) %>% 
  unlist() %>% 
  mean() #0.882, 0.3113267

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByC, lowPerformanceSampleByQ) %>% 
  unlist()

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByC, lowPerformanceSampleByQ, FALSE) %>% 
  unlist() %>% 
  mean() #0.8475, 0.2128282

#import the ground truths
trueLabels <- read_tsv("../../touche23/labels-training.tsv")

randomSampleLabels <- aRandomSampleByQ %>% 
  select("Argument ID") %>% 
  left_join(trueLabels)

lowPerformanceSampleLabels <- lowPerformanceSampleByQ %>% 
  select("Argument ID") %>% 
  left_join(trueLabels)

#compare between Q and truth: 0.8485, 0.78; 0.2928231, 0.2031875
lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByQ, randomSampleLabels) %>% 
  unlist() 

lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByQ, randomSampleLabels, FALSE) %>% 
  unlist() %>% 
  mean()

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByQ, lowPerformanceSampleLabels) %>% 
  unlist()

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByQ, lowPerformanceSampleLabels, FALSE) %>% 
  unlist() %>% 
  mean()

#compare between C and truth: 0.8465, 0.7775; 0.2853836, 0.186113
lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByC, randomSampleLabels) %>% 
  unlist()

lapply(valueNames, computeKrippendorffsAlpha, aRandomSampleByC, randomSampleLabels, FALSE) %>% 
  unlist() %>% 
  mean()

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByC, lowPerformanceSampleLabels) %>% 
  unlist() 

lapply(valueNames, computeKrippendorffsAlpha, lowPerformanceSampleByC, lowPerformanceSampleLabels, FALSE) %>% 
  unlist() %>% 
  mean()

# where the difference occurs

matrix1 <- as.matrix(select(aRandomSampleByQ, -`Argument ID`, -Conclusion, -Stance, -Premise, -coder))

matrix2 <- as.matrix(select(randomSampleLabels, -`Argument ID`))

compareTruthVsQSimpleRandom <- matrix1 == matrix2 
compareTruthVsQSimpleRandom %>% 
  as_tibble() %>% 
  write_csv2("compareTruthVsQSimpleRandom.csv")


library(corrplot)

res <- trueLabels %>% 
  select(-'Argument ID') %>% 
  cor()

corrplot(res, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)
