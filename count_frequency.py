import json
from collections import defaultdict

def count_intent(data):
    intent_count =  defaultdict(int)
    total = 0
    for entry in data:
        intent_count[entry['intent']] += 1
        total+=1
    return intent_count, total

train_data = json.load(open('train_data.json'))
test_data = json.load(open('test_data.json'))

intent_train, total_train = count_intent(train_data)
intent_test, total_test = count_intent(test_data)

for k in sorted(intent_train.keys()):
    print(k, intent_train[k]/total_train)
for k in sorted(intent_test.keys()):
    print(k, intent_test[k]/total_test)

# print(count_intent(train_data))
# print(count_intent(test_data))
