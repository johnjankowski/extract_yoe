Classifiers to find years of experience from pdf resumes.

gen_resumes.py generates some sample resumes to train the classifier on and saves them to text.p and labels.p

classify.py trains models and uses them to predict or loads a model and predicts.
In classify.py you must uncomment the lines under either 'CRF script' or 'random forest script' to choose the classifier you want.
To use you can call it from the command line as "python crf_classify.py [Directory with txt files]"
It will print out what it extracts to stdout in the format "resume line | word labels"
