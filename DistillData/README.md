# Distill data

A small python project for distill emotion related data from VAD-Lexicon

## Introduction

Since NRC-VAD-Lexicon has VAD value for every kind of words, if all of these words are Implemented, AI will be confused predicting the virtual persona's emotion. Thus, this code will distill only emotion words(ex. happiness) from the Lexicon.


## How

1. Since this distilled data needs high-quality, it will use OpenAI API call to ask whether this word is emtion or not.
2. Parse original dataset (.txt) to array of json.
3. As output, it will dump another array of json only with emotion.
4. Run this code with dockfile in GCP vm

