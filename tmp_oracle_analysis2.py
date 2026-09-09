import pandas as pd, numpy as np

subtype = pd.read_csv('runs/predict_gene_expression_20260907_162010/val_per_gene_metrics.csv').set_index('gene')
twohead = pd.read_csv('runs/predict_gene_expression_20260908_093459/val_per_gene_metrics.csv').set_index('gene')

merged = pd.DataFrame({'subtype': subtype.r2, 'two_head': twohead.r2}).dropna()
oracle = merged.max(axis=1)
print('subtype-only mean R2:  ', merged.subtype.mean())
print('two_head mean R2:      ', merged.two_head.mean())
print('oracle (best-of) mean R2:', oracle.mean())
print()
print('genes where two_head > subtype:', (merged.two_head > merged.subtype).sum(), '/', len(merged))
print('genes where subtype > two_head:', (merged.subtype > merged.two_head).sum(), '/', len(merged))
print()
delta = merged.two_head - merged.subtype
print('Top 10 genes most helped by two_head vs subtype-only:')
print(delta.sort_values(ascending=False).head(10))
print()
print('Top 10 genes most hurt by two_head vs subtype-only:')
print(delta.sort_values().head(10))
print()
print('correlation subtype vs two_head per-gene R2:', merged.subtype.corr(merged.two_head))
