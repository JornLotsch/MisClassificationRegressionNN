#################################### Libraries ########################################################################

library(parallel)
library(ggplot2)
library(ggpubr)
library(ggforce)
library(cowplot)
library(caret)
library(gridGraphics)
library(grid)
library(gridExtra)
library(neuralnet)

nProc <- detectCores() - 1 # round((detectCores() - 1) / 4) # /4 RF and MultiOmics data

nIter <- 100
seed <- 42
list.of.seeds <- 1:nIter + seed - 1

#################################### Get data ########################################################################
DataSets <- c("FourGauss", "FourGaussSimple")
architectures <- list(c(0), c(1), c(2), c(3)) # Neural net architectures for nnet

for (ActualDataSet in DataSets) {
  switch(ActualDataSet,
    FourGauss = {
      ActualData <- read.csv("GaussXYin2Classes.csv", row.names = 1)
    },
    FourGaussSimple = {
      ActualData <- read.csv("GaussXYin2Classes.csv", row.names = 1)
      ActualData$Cls <- ifelse(ActualData$X < 0, 1, 2)
    }
  )

  # All pre
  dupl <- which(duplicated(within(ActualData, rm(Cls))))
  which(duplicated(within(ActualData, rm(Cls)), fromLast = T))
  if (!identical(integer(0), dupl) == TRUE) {
    ActualData <- ActualData[-dupl, ]
  }
  sapply(ActualData, as.numeric)
  sum(apply(ActualData, 2, function(x) length(which(is.na(x)))))
  table(ActualData$Cls)

  #################################### Split data into training/test validation ########################################################################

  ValidationSampleFraction <- 0.8

  if (ValidationSampleFraction != 0) {
    SplittedData <- opdisDownsampling::opdisDownsampling(
      Data = within(ActualData, rm(Cls)), Cls = ActualData$Cls,
      Size = ValidationSampleFraction * nrow(ActualData),
      Seed = 42, nTrials = 10000, MaxCores = nProc # , JobSize = 100
    )
    TrainingTestData <- SplittedData$ReducedData
    ValidationData <- SplittedData$RemovedData
    print(dim(TrainingTestData))
    print(dim(ValidationData))
  } else {
    TrainingTestData <- ActualData
    ValidationData <- NULL
  }

  TrainingTestData$Cls <- make.names(TrainingTestData$Cls)
  ValidationData$Cls <- make.names(ValidationData$Cls)

  #################################### Train and validate neural nets ########################################################################

  BAArch <- lapply(architectures, function(arch) {
    model <- neuralnet(
      as.factor(Cls) ~ .,
      data = TrainingTestData,
      hidden = arch,
      act.fct = "logistic",
      # threshold = 0.001,
      rep = 5,
      linear.output = FALSE
    )

    pic <- grid.grabExpr(
      plot(model,
        rep = "best", information = F,
        col.hidden = "darkblue",
        intercept = F,
        col.entry = "darkgreen",
        col.out = "darkgrey", show.weights = F
      )
    )

    ClassificationResultsNN <- pbmcapply::pbmclapply(list.of.seeds, function(x) {
      set.seed(x)

      inTraining <- createDataPartition(TrainingTestData$Cls, p = .67, list = FALSE)
      training <- TrainingTestData[inTraining, ]
      inValidation <- createDataPartition(ValidationData$Cls, p = .8, list = FALSE)
      validation <- ValidationData[-inValidation, ]

      model <- neuralnet(
        as.factor(Cls) ~ .,
        data = training,
        hidden = arch,
        act.fct = "logistic",
        # threshold = 0.001,
        rep = 5,
        linear.output = FALSE
      )

      pred <- data.frame(predict(model, validation))
      pred <- apply(pred, 1, function(x) unique(validation$Cls)[which.max(x)])
      CM <- confusionMatrix(
        factor(pred, levels = unique(validation$Cls)),
        factor(validation$Cls, levels = unique(validation$Cls))
      )

      if (length(unique(validation$Cls)) > 2) {
        BA <- mean(CMnn$byClass[, "Balanced Accuracy"])
      } else {
        BA <- CM$byClass["Balanced Accuracy"]
      }
      return(BA)
    }, mc.cores = nProc)

    # BAmean <- mean(unlist(ClassificationResultsNN))
    return(list(ClassificationResultsNN = ClassificationResultsNN, pic = pic))
  })


  names(BAArch) <- make.names(paste0("N", architectures))
  dfBAs <- do.call(cbind.data.frame, lapply(names(BAArch), function(x) unlist(BAArch[[x]]$ClassificationResultsNN)))
  names(dfBAs) <- names(BAArch)


  ClassificationResultsReg <- pbmcapply::pbmclapply(list.of.seeds, function(x) {
    set.seed(x)

    inTraining <- createDataPartition(TrainingTestData$Cls, p = .67, list = FALSE)
    training <- TrainingTestData[inTraining, ]
    inValidation <- createDataPartition(ValidationData$Cls, p = .8, list = FALSE)
    validation <- ValidationData[-inValidation, ]

    training$Cls <- as.integer(as.factor(training$Cls)) - 1
    validation$Cls <- as.integer(as.factor(validation$Cls)) - 1

    RegModel <- glm(as.factor(Cls) ~ ., family = binomial(link = "logit"), data = as.data.frame(training))
    pred <- predict(RegModel, newdata = validation, type = "response")
    pred <- ifelse(pred < 0.5, unique(validation$Cls)[1], unique(validation$Cls)[2])

    CM <- confusionMatrix(
      factor(pred, levels = unique(validation$Cls)),
      factor(validation$Cls, levels = unique(validation$Cls))
    )

    if (length(unique(validation$Cls)) > 2) {
      BA <- mean(CM$byClass[, "Balanced Accuracy"])
    } else {
      BA <- CM$byClass["Balanced Accuracy"]
    }
    return(BA)
  }, mc.cores = nProc)

  dfBAs["Regression"] <- unlist(ClassificationResultsReg)
  dfBAs_long <- reshape2::melt(dfBAs)

  assign(
    paste0("pBAs_", ActualDataSet),
    ggplot(data = dfBAs_long, aes(x = variable, y = value, group = variable)) +
      geom_boxplot(fill = "cornsilk", outlier.shape = NA) +
      geom_sina() +
      labs(title = "Balanced accuracy (100-fold CV)", x = "Model", y = "BA") +
      theme_light() +
      ylim(0, 1)
  )
  assign(
    paste0("pActualData_", ActualDataSet),
    ggplot(data = ActualData, aes(x = ActualData[, 1], y = ActualData[, 2], color = factor(Cls))) +
      geom_point(show.legend = F) +
      coord_fixed() +
      coord_equal() +
      scale_color_manual(values = c("chartreuse3", "dodgerblue")) +
      labs(title = "Original data (2 classes)") +
      theme_light()
  )

  assign(
    paste0("pNN_", ActualDataSet),
    plot_grid(
      plot_grid(
        BAArch$N0$pic,
        BAArch$N1$pic,
        BAArch$N2$pic,
        BAArch$N3$pic,
        labels = LETTERS[1:4],
        nrow = 1,
        align = "v", axis = "tb"
      ),
      plot_grid(
        get(paste0("pActualData_", ActualDataSet)),
        get(paste0("pBAs_", ActualDataSet)),
        labels = LETTERS[5:6],
        align = "v", axis = "tb",
        nrow = 1
      ),
      nrow = 2
    )
  )
  print(get(paste0("pNN_", ActualDataSet)))
}

plot_grid(
  plot_grid(
    BAArch$N0$pic,
    BAArch$N1$pic,
    BAArch$N2$pic,
    BAArch$N3$pic,
    labels = LETTERS[1:4],
    nrow = 1,
    align = "v", axis = "tb"
  ),
  plot_grid(
    get(paste0("pActualData_FourGauss")),
    get(paste0("pBAs_FourGauss")),
    labels = LETTERS[5:6],
    align = "v", axis = "tb",
    nrow = 1
  ),
  plot_grid(
    get(paste0("pActualData_FourGaussSimple")),
    get(paste0("pBAs_FourGaussSimple")),
    labels = LETTERS[7:8],
    align = "v", axis = "tb",
    nrow = 1
  ),
  nrow = 3
)
