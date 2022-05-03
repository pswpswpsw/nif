import pandas as pd
import tfrecorder

csv_file = '/path/to/images.csv'
df = pd.read_csv(csv_file, names=['split', 'image_uri', 'label'])
df.tensorflow.to_tfr(output_dir='/my/output/path')
