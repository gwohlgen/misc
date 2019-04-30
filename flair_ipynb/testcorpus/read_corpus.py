import codecs

for line in codecs.open('corpus.csv', 'r', 'utf-8', errors='replace'):
    line = line.strip()
    parts = line.split(',')

    label = parts[-1]
    text = ','.join( parts[:-1] )

    print(label, text)
   
