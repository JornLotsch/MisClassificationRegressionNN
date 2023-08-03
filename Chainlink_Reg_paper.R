# Lotsch and Ultsch, 2023
# For Windows system, please replace "pbmcapply::pbmclapply" with "lapply"
# The speed will be slow on Windows systems, or the parallel processing needs to be re-implemented in a Windows compatible manner

#################################### Libraries ########################################################################

library(FCPS)
library(opdisDownsampling)

#################################### Get data ########################################################################

ActualData <- cbind.data.frame(FCPS::Chainlink$Data, Cls = FCPS::Chainlink$Cls)

ValidationSampleFraction <- 0.8
if (ValidationSampleFraction != 0) {
  SplittedData <- opdisDownsampling::opdisDownsampling(
    Data = within(ActualData, rm(Cls)), Cls = ActualData$Cls,
    Size = ValidationSampleFraction * nrow(ActualData),
    Seed = 42, nTrials = 10000
  )
  TrainingTestData <- SplittedData$ReducedData
  ValidationData <- SplittedData$RemovedData
  print(dim(TrainingTestData))
  print(dim(ValidationData))
} else {
  TrainingTestData <- ActualData
  ValidationData <- NULL
}

#################################### Perform regression ########################################################################

TrainingTestData_preProcessed <- TrainingTestData
TrainingTestData_preProcessed[names(TrainingTestData_preProcessed) != "Cls"] <- apply(within(TrainingTestData_preProcessed, rm(Cls)), 2, scale)
Reg <- glm(as.factor(Cls) ~ ., family = binomial(link = "logit"), data = as.data.frame(TrainingTestData_preProcessed))
print(summary(Reg))


