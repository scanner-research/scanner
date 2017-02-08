from scannerpy import Database

db = Database()
sampler = db.sampler()

# To string together multiple ops into a pipeline, you can pass a list of ops to
# db.run and they will be run in order. Here, we first blur each video frame
# then compute its color histograms.
blur = db.ops.Blur()
hist = db.ops.Histogram()
tasks = sampler.all([('test', 'test_hist_blurred')])
db.run(input_table, [blur, hist])

# With pipelines like the one above, all the outputs from the Blur op are passed
# inputs to the Histogram op. However, if you want to customize your inputs and
# outputs, or if you want to pass values between ops not adjacent in a pipeline,
# you can explicitly list op inputs.
