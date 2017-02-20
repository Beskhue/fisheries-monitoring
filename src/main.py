from clize import run
import pipeline
import pprint

def example():
    """
    Run the pipeline example (tests if the pipeline runs succesfully, should output ).
    """

    pl = pipeline.Pipeline()

    a = pl.train_and_validation_mini_batch_generator_generator()
    train_mini_batches = a['train']

    x, y, meta = next(train_mini_batches)
    print("Number of cases in first batch: %s" % len(x))
    print("First image shape and label: %s - %s" % (str(x[0]().shape), y[0]))
    print("First image meta information:")
    pprint.pprint(meta[0])

if __name__ == "__main__":
    run(example)
