AI2 Reasoning Challenge (February 2018)


*****
* Do not distribute *
* Non-commercial use only *

This data should not be distributed except by the Allen Institute for Artificial Intelligence (AI2). All parties interested in acquiring this data must download it from AI2 directly at data.allenai.org/arc. This data is to be used for non-commercial, research purposes only. 

Please contact arc@allenai.org with any questions regarding this dataset. Please visit allenai.org/data for a complete list of AI2’s datasets.
*****


About: ARC Science Questions
=====
This data set consists of 7787 science exam questions drawn from a variety of sources, including science questions provided under license by a research partner affiliated with AI2. These are text-only, English language exam questions that span several grade levels as indicated in the files. Each question has a multiple choice structure (typically 4 answer options).
The questions are sorted into a Challenge Set of 2590 “hard” questions (those that both a retrieval and a co-occurrence method fail to answer correctly) and an Easy Set of 5197 questions. Each are pre-split into Train, Development, and Test sets. Each set is provided in two formats, CSV and JSON. The CSV files contain the full text of the question and its answer options in one cell. The JSON files contain a split version of the question, where the question text has been separated from the answer options programatically.

The question counts are as follows:

Challenge Train: 1119
Challenge Dev: 299
Challenge Test: 1172
Easy Train: 2251
Easy Dev: 570
Easy Test: 2376


Columns of the CSV
——
questionID: Unique identifier for the question.
originalQuestionID: Legacy ID used within AI2.
totalPossiblePoint: The point value of the question for grading purposes.
AnswerKey: The letter signifying the correct answer option for the question.
isMultipleChoice: 1 indicates the question is multiple choice.
includesDiagram: 0 indicates the question does not include a diagram.
examName: The name of the source exam for these questions.
schoolGrade: The intended grade level for the question.
year: The year the questions were sourced for AI2.
question: The question and its answer options. Each answer option is indicated by a letter in parentheses, e.g., (A) and (B).
subject: The question's subject; this is left blank in this data set.
category: Whether the question is a Train, Dev, or Test question.


Structure of the JSON
——
The JSON files contain the same questions split into the "stem" of the question (the question text) and then the various answer "choices" and their corresponding labels (A, B, C, D). The questionID is also included.


About: ARC Corpus
=====

The ARC Corpus contains 14M unordered, science-related sentences including knowledge
relevant to ARC, and is provided to as a starting point for addressing the challenge. 

This dataset was built from documents publicly available through the major search engines, from dictionary definitions from Wiktionary, and from articles from Simple Wikipedia that were tagged as science. For further details of its construction, see (Clark et al., 2018). 

Note that use of the corpus for the Challenge is completely optional, and also that systems are not restricted to this corpus. 

Terms of use of this corpus:
This corpus was created in support of the development of a scientific and research focused AI system. This dataset is provided to you solely for non-commercial research and educational purposes. You may not extract individual documents from this dataset or use this dataset for any other purpose. You may not redistribute this dataset to others. THIS DATASET IS PROVIDED WITH NO WARRANTY, EITHER EXPRESSED OR IMPLIED, AND AI2 EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR LACK OF THIRD-PARTY INFRINGEMENT. Use at your own risk. You agree to indemnify AI2 for any claims arising out of your use of this dataset.


About the acquisition and format of this corpus:
The majority of the data in this corpus was collected by utilizing a major search engine to run a series of search terms that are relevant to the science concepts about which the Aristo project is collecting knowledge. The top several documents from each search result list were collected and deduplicated, and then the content of these documents was stripped down to capture just the text in each document. The resulting text was then chunked into sentences, cleaned of formatting errors, and randomly sorted. We are presenting this corpus as one flat file of sentences compiled from all of these collected documents. No other markup has been included.

About the research goals of this dataset:
This dataset is intended to provide a corpus of English language sentences about science topics. The topics included are based on the science curriculum covered by US elementary and middle schools, with the intent to support Project Aristo in its research into the necessary natural language processing to both understand and answer student science assessment questions. It is our hope that this dataset provides a useful and novel corpus to non-commercial researchers seeking this type of data for their own NLP, machine reading, or other related efforts.
