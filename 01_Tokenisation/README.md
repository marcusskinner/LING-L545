I used two different segmenters written in Python. The first one, using the package Spacy, tokenizes the text using a set of rules specific to a language. I'm not sure what those rules are, but the docs says it takes into account contractions, abbreviations and other anomalies that might be in the language. The second is my custom segmenter that just splits each sentence by a period. Obviously this is not going to account for every case, but I'm curious how a barebones segmenter will compare to an industrial-grade segmenter.

<h4>How should you segment sentences with semi-colon? As a single sentence or as two sentences? Should it depend on context?</h4>
In english, I'd say no since a semicolon doesn't terminate a sentence. But there might be lenguages where a semicolon does terminate a sentence. So, it'd depend on the language.

<h4>Should sentences with ellipsis... be treated as a single sentence or as several sentences?</h4>
I'd say it depends on the context.

<h4>If there is an exclamation after the first word in the sentence should it be a separate sentence? How about if there is a comma?</h4>
A exclamation point can sometimes just be an interjection. In that context, I'd say no. But it can also be a sentence terminator. So, it depends. In english, I'd say a comma doesn't terminate a sentence, but there might be a language where it does.

<h4>Can you think of some hard tasks for the segmenter</h4>
The segmenter is going to have a hard time with multiple different languages. It's also going to have a hard time with dialogue since quotes can contain multiple sentences but are still part of a single sentence. Also abbreviations or sentence terminators that are used in a context that does not terminate a sentence will be hard. Also typos and grammatical mistakes might also throw the segmentor off.

<h4>Why should we split punctuation from the token it goes with ?</h4>
Sometimes if we want to consider the word by itself, it might be harder with the punctuation symbol. 

<h4>Should abbreviations with space in them be written as a single token or two tokens ?</h4>
I think it might depend on what we're doing. Usually, I'd say an appreviation should be its own token because it represents a single word or phrase. With numerals, I think it'd also depend on what we're trying to do.

<h4>If you have a case suffix following punctuation, how should it be tokenised?</h4>

<h4>Should contractions and clitics be a single token or two (or more) tokens?</h4>
I'd say contractions should be their own token. But again, I think it depends on what we're trying to do.