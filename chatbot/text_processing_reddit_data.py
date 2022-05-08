import os
import codecs
import re
import csv

"""
This file is dedicate to preprocessing the raw data reddit into a text file full of sequential conversations.
"""

datasetName = "custom reddit-dialogs dataset"
dataset = os.path.join("chatbot\data", datasetName)

parentData = os.path.join(dataset, "reddit_parent_reply.txt")
childData = os.path.join(dataset, "reddit_child_reply.txt")

question = []
reply = []

with open(parentData, encoding='iso-8859-1') as parent:
    with open(childData, encoding='iso-8859-1') as child:
        with open("chatbot/test.txt", "w", encoding="utf-8") as newFile:
            question = parent.readlines()
            reply = child.readlines()

            for line in question, reply:
                line = re.sub(
                    r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', "", str(line))
                line = re.sub("newlinechar", "", str(line))

            for i in range(len(question)):
                line = question[i].strip() + "\n" + reply[i]
                newFile.write(line)
