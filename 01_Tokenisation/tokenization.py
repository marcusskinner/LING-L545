import sys

text = sys.stdin.read()

spec_chars = [",", ":", "(", ")", ".", "!", "?"]

# put a space before and after special characters
for c in spec_chars:
    text = text.replace(c, " " + c + " ")

# Gets rid of extra spacing. Can be increased if more extra spacing is found
for i in range(1, 4):
    text = text.replace("  ", " ")
    
# replace every space with a newline character
text = text.replace(" ", "\n")

sys.stdout.write(text)