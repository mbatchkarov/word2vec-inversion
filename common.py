from zipfile import ZipFile
import json, re



# cleaner (order matters)
def clean(text): 
    contractions = re.compile(r"'|-|\"")
    # all non alphanumeric
    symbols = re.compile(r'(\W+)', re.U)
    # single character removal
    singles = re.compile(r'(\s\S\s)', re.I|re.U)
    # separators (any whitespace)
    seps = re.compile(r'\s+')

    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

# sentence splitter
alteos = re.compile(r'([!\?])')
def sentences(l):
    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
    return l.split(".")


def YelpReviews(label):
    with ZipFile("yelp_%s_set.zip"%label, 'r') as zf:
        with zf.open("yelp_%s_set/yelp_%s_set_review.json"%(label,label)) as f:
            for i, line in enumerate(f):
                rev = json.loads(line.decode())
                yield {'y':rev['stars'],\
                       'x':[clean(s).split() for s in sentences(rev['text'])]}

def StarSentences(reviews, stars=[1,2,3,4,5]):
    for r in reviews:
        if r['y'] in stars:
            for s in r['x']:
                yield s