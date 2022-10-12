<h1>Tagger Comparison</h1>

<h2>UDPipe Tagger</h2>

Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------|-----------|-----------|-----------|-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |     94.77 |     94.77 |     94.77 |     94.77
XPOS       |     95.96 |     95.96 |     95.96 |     95.96
Feats      |     91.04 |     91.04 |     91.04 |     91.04
AllTags    |     90.04 |     90.04 |     90.04 |     90.04
Lemmas     |     84.94 |     84.94 |     84.94 |     84.94
UAS        |    100.00 |    100.00 |    100.00 |    100.00
LAS        |    100.00 |    100.00 |    100.00 |    100.00

<h2>Perceptron Tagger</h2>

Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------|-----------|-----------|-----------|-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |     90.58 |     90.58 |     90.58 |     90.58
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
Feats      |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |     90.58 |     90.58 |     90.58 |     90.58
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |    100.00 |    100.00 |    100.00 |    100.00
LAS        |    100.00 |    100.00 |    100.00 |    100.00

<h1>Improved Perceptron Tracker</h1>

<h2>Before</h2>

Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------|-----------|-----------|-----------|-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |     96.20 |     96.20 |     96.20 |     96.20
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
Feats      |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |     96.20 |     96.20 |     96.20 |     96.20
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |    100.00 |    100.00 |    100.00 |    100.00
LAS        |    100.00 |    100.00 |    100.00 |    100.00

<h2>After</h2>

I created a script (optimize.py) that trained the perceptron with a random subset of features by modifying the tagger.py file to take in a set of flags. If a flag is set to 1, then the perceptron will consider that feature. The highest I achieved was 96.39 accuracy by removing the features "bias," "i pref1," "i-1 word" and "i+1 suffix"

Metrics    | Precision |    Recall |  F1 Score | AligndAcc
-----------|-----------|-----------|-----------|-----------
Tokens     |    100.00 |    100.00 |    100.00 |
Sentences  |    100.00 |    100.00 |    100.00 |
Words      |    100.00 |    100.00 |    100.00 |
UPOS       |     96.39 |     96.39 |     96.39 |     96.39
XPOS       |    100.00 |    100.00 |    100.00 |    100.00
Feats      |    100.00 |    100.00 |    100.00 |    100.00
AllTags    |     96.39 |     96.39 |     96.39 |     96.39
Lemmas     |    100.00 |    100.00 |    100.00 |    100.00
UAS        |    100.00 |    100.00 |    100.00 |    100.00
LAS        |    100.00 |    100.00 |    100.00 |    100.00