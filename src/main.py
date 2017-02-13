import pipeline

pl = pipeline.Pipeline()

a = pl.train_and_validation_mini_batch_generator_generator()
train_mini_batches = a['train']

for x, y, meta in train_mini_batches:
    print(len(x), len(y))
