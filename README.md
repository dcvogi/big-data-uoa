# University of Athens - Big Data Assignment 2018-2019

## Big Data Mining Techniques (M118)
## Winter Semester 2018-2019

Assignment Description included in *desc* folder

### The objective of the exercise

The purpose of the work is to familiarize you with the basic stages of the process used to
implement data mining techniques, namely: collection, preprocessing, cleaning, conversion,
application of data mining techniques and evaluation. The implementation will be in the
Python programming language using the SciKit Learn tool and the gensim library.

### Description

The work is related to the classification of text data from news articles. The Dataset consists
of CSV files whose fields are separated by the '\ t' (TAB) character. Two files are included:

    1. train_set.csv (12267 data points): This file will be used to train your algorithms and contains the following fields:
        a. Id: A unique number for each document
        b. Title: The title of the article
        c. Content:The content of the article
        d. Category: The category of each document
    2. test_set.csv (3068 data points): This file will be used to make predictions for new data points. The CSV file contains all fields of the training file except the 'Category' field. You have to estimate this using classification algorithms.
        
The article categories are 5 and are presented below:

    1. Business
    2. Film
    3. Football
    4. Politics
    5. Technology

### WordCloud Creation
At this point you have to create a Wordcloud for the five categories of articles with the most
articles. To create a WordCloud for a particular category you will use all the articles of this
category. An example of a WordCloud is shown in the following figure. You can use any
Python library in order to create the WordClouds.

### Duplicates Detection
Here you should find similar articles. In particular, the similarity between two articles will be
measured using the cosine similarity between the term vectors of each article. Anyone who
wants can use the LSH technique in order to quickly identify duplicates. Your code should
accept a similarity threshold θ. Finally, all pairs of text with a similarity greater than 0.7
should be reported. The results will be stored in the file ‘duplicatePairs.csv’ and will have the
following format:


### Υλοποίηση Κατηγοριοποίησης (Classification)
Here you have to test 2 classification Classification techniques:
    * Support Vector Machines (SVM).
    * Random Forests.
Also, you have to evaluate the performance of the above classification techniques using the
following features:
    * Bag of Words (BoW).
    * SVD keeping the 90% of the total variance (SVD).
    * Average word vector for each vector (W2V).
    
You should also evaluate and report the performance of each method using 10-fold Cross
Validation using the following metrics:
    * Accuracy
    * Precision
    * Recall
    
    
### Beat the Benchmark
Finally, you should experiment with any Classification technique or approach you want, by
doing any pre-processing to the data you want to overcome as much as possible the results
achieved at your previous query. You should report and justify the steps you have taken.

Output Files
Your code should for the queries related to Classification should create the following files:
    * EvaluationMetric_10fold.csv
    * testSet_categories.csv
    * roc_10fold.png
    
### Regarding the Report
The folder that you will submit should have the name:
Ass1_name1_studentID1_name2_studentID2.
The folder will contain:
    1. A text report that explains your experiments and the methods that you tested in PDF
    format. Your report should also contain the requested tables that describe the
    performance of tje tested methods and your report should not exceed 30 pages.
    2. The requested output files.
    3. The Python code files that you wrote.
    
The extensive report that you will deliver should contain the description for your tests and
anything else you think to show the tests that you have done, and explain why the chosen
methods have the reported results. All work will be evaluated on the basis of the correct
documentation and to the extent that they satisfy the demands of this exercise.

Useful Links:
    * https://radimrehurek.com/gensim/models/word2vec.html
    * https://code.google.com/archive/p/word2vec/
    * https://radimrehurek.com/gensim/models/ldamodel.html
