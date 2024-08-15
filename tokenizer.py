import re

with open("TrainningData/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab = {token:i for i,token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i > 50: break
