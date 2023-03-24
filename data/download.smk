import pandas as pd

manifest = pd.read_csv('manifest_230324.csv')

# Randomly sample 10% of the dataset.
manifest = manifest.sample(frac=0.1, random_state=42)

f2type = {r.filename:r.seq_type for r in manifest.to_records()}
f2study = {r.filename:r.study for r in manifest.to_records()}

filenames = manifest.filename.values
ALL = expand('sequences/{filename}.list', filename=filenames)

rule all:
    input: ALL

rule download:
    output:
        'sequences/{filename}.list'
    params:
        type = lambda wc: f2type[wc.filename],
        study = lambda wc: f2study[wc.filename],
    shell:
        'wget -qO- '
        'http://opig.stats.ox.ac.uk/webapps/ngsdb/{params.type}/{params.study}/csv/{wildcards.filename}.csv.gz | '
        'gunzip -c | '
        'tail -n+3 | '
        'cut -d, -f1 > {output}'
