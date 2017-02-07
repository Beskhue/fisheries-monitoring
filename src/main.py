import pipeline

pl = pipeline.Pipeline()

a = pl.trainAndValidationMiniBatchGeneratorGenerator()
trainMiniBatches = a['train']

for x, y in trainMiniBatches:
    print(len(x), len(y))
