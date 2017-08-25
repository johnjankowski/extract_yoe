

f = open('raw_resumes.txt', 'r')
text = f.read().split()
f.close()
f = open('resume_words.csv', 'w')
f.write('Word' + '\n')
for word in text:
    if ',' in word:
        word = word.replace(',', '')
    for i in range(len(word)):
        try:
            word[i].encode('utf-8')
        except UnicodeError:
            word = word.replace(word[i], ' ')
    word = word.replace(' ', '')
    f.write((word + '\n').encode('utf-8'))
f.close()