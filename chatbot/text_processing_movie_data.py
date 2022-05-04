import os
import codecs
import re
import csv

"""
This file is dedicate to preprocessing the raw data (cornel movie-dialogs corpus) into a text file full of sequential conversations.
"""

datasetName = "cornell movie-dialogs dataset"

# Where the dataset is placed
dataset = os.path.join("chatbot\data", datasetName)


def loadLines(fileName, params):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(params):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups params of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, params):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract params
            convObj = {}
            for i, field in enumerate(params):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        # We ignore the last line (no answer for it)
        for i in range(len(conversation["lines"]) - 1):
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Save the new datasetName as a .txt file to this directory
datafile = os.path.join(dataset, "formatted_movie_lines.txt")

delimiter = '\t'  # We require a delimiter and a way to escape from it
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}  # Dictionary correlation between line id's with actual text
conversations = []  # Contains conversations
MOVIE_LINES_params = ["lineID", "characterID",
                      "movieID", "character", "text"]  # params in raw data.
MOVIE_CONVERSATIONS_params = ["character1ID",
                              "character2ID", "movieID", "utteranceIDs"]

lines = loadLines(os.path.join(dataset, "movie_lines.txt"), MOVIE_LINES_params)
conversations = loadConversations(os.path.join(dataset, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_params)

print("\nNEW FILE CREATED")
with open(datafile, 'w', encoding='utf-8') as outputfile:  # Write to the file using CSV module
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)
