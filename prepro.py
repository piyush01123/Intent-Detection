import json

def tokenize_line(line):
    sentence, label = line.split('\t')
    sentence = sentence[4:-4]   #get rid of BOS and EOS terms
    words_, label_ = sentence.split(' '), label[:-1].split(' ') #get rid of newline
    intent_ = label_.pop()
    intent_ = intent_[5:] #remove the phrase atis_
    if '#' in intent_:
        intent_ = intent_.split('#')[0]
    intents.append(intent_)
    label_ = label_[1:]
    subintents = {}
    for i, l in enumerate(label_):
        if l =='O':
            pass
        else:
            subintents[l] = words_[i]
    return {'sentence': sentence,
            'intent': intent_,
            'subintents': subintents
            }

if __name__=='__main__':
    train_lines = open('atis-2.train.w-intent.iob (3).txt', 'r').readlines()
    test_lines = open('atis.test.w-intent.iob (2).txt', 'r').readlines()
    train_data = []
    test_data = []
    intents = []
    for line in train_lines:
        train_data.append(tokenize_line(line))
    with open('train_data.json', 'w') as outfile:
        json.dump(train_data, outfile)
    for line in test_lines:
        test_data.append(tokenize_line(line))
    with open('test_data.json', 'w') as outfile:
        json.dump(test_data, outfile)
    with open('intent_list.txt', 'a+') as f:
        for intent in list(set(intents)):
            f.write(intent+'\n')
