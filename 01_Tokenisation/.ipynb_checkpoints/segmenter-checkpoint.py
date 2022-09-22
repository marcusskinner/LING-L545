import sys

text = sys.stdin.read()
text = text.replace(". ", "\n")
sys.stdout.write(text)