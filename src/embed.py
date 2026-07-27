import pandas as pd
import numpy as np
import sys
import time
import torch
import math
from transformers import AutoTokenizer, AutoModel

# Prevent Python from generating .pyc files.
sys.dont_write_bytecode = True

# Project-specific configuration and utility functions.
from config import *
from utils  import *

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

# Printing header.
printHeader("Embedding HPO Terms")

# To track time.
startTime = time.time()







# ------------------------------------------------------------------------------
# Load and Filter Human Phenotype Ontology (HPO) from transformed data file.
# ------------------------------------------------------------------------------

# Ensure the input file exists before proceeding.
exitIfFileNotExist(inputFileTask)

data = readCSV(inputFileTask)

# Retrieving HPO IDs.
log("Retrieving IDs...")
hpoIDs = getHPOIDs(data)
log("IDs retrieved.")

# Preparing Labels.
log("Filtering Labels of HPO Concepts...")
labels      = data[(
        data[classColumn] == labelClass
            ) & (
        data[hpoidColumn].isin(hpoIDs)
    )][contentColumn].tolist()
labelIDs    = data[(
        data[classColumn] == labelClass
            ) & (
        data[hpoidColumn].isin(hpoIDs)
    )][hpoidColumn].tolist()
log("Filtering Labels completed.")

# Preparing Synonyms.
log(f"Filtering Synonyms ({', '.join(synonymClasses)})...")
synonyms    = data[(
        data[classColumn].isin(synonymClasses)
            ) & (
        data[hpoidColumn].isin(hpoIDs)
    )][contentColumn].tolist()
synonymIDs  = data[(
        data[classColumn].isin(synonymClasses)
            ) & (
        data[hpoidColumn].isin(hpoIDs)
    )][hpoidColumn].tolist()
log(f"Filtering Synonyms completed.")
# Data retrieval completed.






# ------------------------------------------------------------------------------
# Embed and calculate the similarity of labels with their respective synonyms.
# ------------------------------------------------------------------------------

log("Start to Embed Synonyms and Labels with all Embedding Models.")
embeddingModelsList = list(embeddingModels.keys())
embeddingModelsList.sort()
for modelName in embeddingModelsList:

    index = embeddingModelsList.index(modelName)

    # Set up the embedding model.
    log(f"Set up ({index + 1}/{len(list(embeddingModelsList))}) " \
        f"model and tokenizer {quote(modelName)}...")
    tokenizer = AutoTokenizer.from_pretrained(embeddingModels[modelName], 
        model_max_length = 64)
    model = AutoModel.from_pretrained(embeddingModels[modelName])

    # Put model in evaluation mode
    model.eval()
    log("Model set up.")

    # Tokenize input.
    log("Tokenize Labels...")
    inputs = tokenizer(labels, padding = True, truncation = True, 
        return_tensors = "pt")
    log("Labels tokenized.")    

    # Generate Embeddings of Labels.
    log("Embedding Labels...")
    with torch.no_grad():
        outputs = model(**inputs)
    log("Labels embedded.")

    # Extract Embeddings.
    log("Finalizing embeddings of Labels...")
    tokenEmbeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(
        tokenEmbeddings.size()).float()
    embeddings = torch.sum(tokenEmbeddings * mask, dim = 1) / torch.clamp(
        mask.sum(dim = 1), min = 1e-9)

    resultLabels = {}
    for i, l, e in zip(labelIDs, labels, embeddings):
        # Since there is just one label per concept there is no need to take
        # into account the label's text.
        resultLabels[i] = e
    log("Finalized Label embeddings.")

    # Tokenize input.
    log("Tokenize Synonyms...")
    inputs = tokenizer(synonyms, padding = True, truncation = True,
        return_tensors="pt")
    log("Synonyms tokenized.")

    # Generate Embeddings of Synonyms.
    log("Embedding Synonyms...")
    with torch.no_grad():
        outputs = model(**inputs)
    log("Synonyms embedded.")

    # Extract Embeddings.
    log("Finalizing embeddings of Synoyms...")
    tokenEmbeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(
        tokenEmbeddings.size()).float()
    embeddings = torch.sum(tokenEmbeddings * mask, dim=1) / torch.clamp(
        mask.sum(dim=1), min=1e-9)

    resultSynonyms = {}
    for i, s, e in zip(synonymIDs, synonyms, embeddings):
        # Since there is the possibility, that two different concepts can have
        # the same synonym, it is worth to take into account the synonym text
        # and the concept ID. 
        resultSynonyms[str(i) + " " + str(s)] = e
    log("Finalized Synonym embeddings.")

    # Each similarity metric and model will get an own column in which the
    # similarity will be saved.
    for similarityMetric in similarityMetrics:
        similarityColumn = similarityColumnPrefix.format(modelName, 
            similarityMetric)
        # Standard value for no similarity is np.nan.
        data[similarityColumn] = [np.nan] * len(data.index)

    with newProgress() as progress:

        task = newTask(progress, len(data.index), "Calculating Similarity")

        for index, row in data.iterrows():
            # The similarity will be placed in the row with the synonyms only
            # in the respective column. 
            if (row[classColumn] in synonymClasses and 
                row[hpoidColumn] in hpoIDs):
                emb1 = resultSynonyms[str(row[hpoidColumn]) + " " + 
                    str(row[contentColumn])].unsqueeze(0)
                emb2 = resultLabels[row[hpoidColumn]].unsqueeze(0)

                emb1 = emb1[0]
                emb2 = emb2[0]

                # Calculate the Similary Metric.
                for similarityMetric in similarityMetrics:
                    similarityColumn = similarityColumnPrefix.format(modelName, 
                        similarityMetric)

                    if similarityMetric == cosineSimilarity:
                        data.loc[index, similarityColumn] = cosSim(emb1, emb2)
                    elif similarityMetric == euclideanSimilarity:
                        data.loc[index, similarityColumn] = eucSim(emb1, emb2)
                    elif similarityMetric == manhattanSimilarity:
                        data.loc[index, similarityColumn] = manSim(emb1, emb2)
                    elif similarityMetric == angularSimilarity:
                        data.loc[index, similarityColumn] = angSim(emb1, emb2)
                    elif similarityMetric == mahalanobisSimilarity:
                        data.loc[index, similarityColumn] = mahSim(emb1, emb2)
                    else:
                        log("No Similarity function found for metric " \
                            f"{quote(similarityMetric)}")

            progress.update(task, advance = 1)

        progress.refresh()

    m = "mean"
    s = "std"

    # Normalization of the similarity metrics: since every metric has it's own
    # range of values, for every model and metric the values are subtracted by
    # the mean and divided by the standard deviation. With this approach
    # the similarity metrics are comparable. 
    for similarityMetric in similarityMetrics:
        similarityColumn = similarityColumnPrefix.format(modelName, 
            similarityMetric)
        data[similarityColumn] = (
                # Subtraction of the mean.
                data[similarityColumn] - data[similarityColumn].mean()
            ) / data[similarityColumn].std()    # Division by the standard 
                                                # deviation.

        # Some values are being calculated for a first, quick evaluation of 
        # the similarity scores. 
        grouped = data.groupby(classColumn)[similarityColumn].agg([m, s])

        means = pd.Series(grouped[m].tolist()).dropna().tolist()
        stds  = pd.Series(grouped[s].tolist()).dropna().tolist()

        log(f"{modelName} with {similarityMetric} Difference: " \
            f"[{exactSynonymClass} - {relatedSynonymClass}]")
        log(f"Mean's Differnece:               {means[0] - means[1]}")
        log(f"Standard Deviation's Difference: {stds[0] - stds[1]}")
        log(f"{modelName} with {similarityMetric} Values:     " \
            f"[{exactSynonymClass},  {relatedSynonymClass}]")
        log(f"Means:                           {str(means)}")
        log(f"Standard Deviation:              {str(stds)}")

        ssmd = math.fabs(means[0] - means[1]) / math.sqrt(math.fabs(
            math.pow(stds[0], 2) - math.pow(stds[1], 2)))

        log("Strictly standardized mean difference: " + str(ssmd))


# ------------------------------------------------------------------------------
# Persist data to disk.
# ------------------------------------------------------------------------------

writeCSV(data, inputFileTask)






# For time tracking.
minutes         = int((time.time() - startTime) // 60)

# Printing Footer with minutes needed for the job.
printHeader(f"Embedding of HPO Terms completed [Minutes: {minutes}]")