CRF classifier to find years of experience from pdf resumes.

gen_resumes.py generates some sample resumes to train the classifier on and saves them to text.p and labels.p

crf_classify.py trains models and uses them to predict or loads a model from crf_model.p and predicts.
to use as is you can call it from the command line as "python crf_classify.py [Directory with txt files]"
it will print out what it extracts to stdout in the format "resume line | word labels"