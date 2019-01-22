from img_pair_ds.image_pair import ImagePair
import tensorflow as tf
import multiprocessing
class AndData(ImagePair):

    def _key_gen(self, df):
        '''
        Return expression to build label for two images being of same type
        '''
        return (df['file1'].str[0:4] == df['file2'].str[0:4]) * 1    

    def _img_norm(self, img):
        return tf.divide(tf.subtract(tf.cast(255, tf.uint8),img), tf.cast(255,tf.uint8) )

    def _decode(self, f1, f2, fp, flatten=False):
        if not fp[-1]=='/':
            fp+='/'
        img1 = tf.image.decode_png(tf.read_file(fp+f1), 1)
        img1 = self._img_norm(img1)
        img1 = tf.image.resize_image_with_pad(img1, 75, 200, method=tf.image.ResizeMethod.AREA )
        img2 = tf.image.decode_png(tf.read_file(fp+f2), 1)
        img2 = self._img_norm(img2)
        img2 = tf.image.resize_image_with_pad(img2, 75, 200, method=tf.image.ResizeMethod.AREA )
        if flatten:
            img = tf.reshape(tf.concat([img1,img2], axis=0), [-1,1])
        else:
            img = tf.reshape(tf.concat([img1,img2], axis=0), [150,200])
        return img

    def get_tf_batch(self, graph, batch_size=32, flatten=False, prefetch_batches=2, dataset='train', skip_header_lines=1, shuffle_buffer=100000, csv_delimiter=',', ):
        assert dataset.lower() in ['train', 'tr','validation', 'valid', 'va', 'test', 'ts', 'te'], 'dataset must be either train, test, or validation'
        if dataset.lower() in ['train', 'tr']:
            csv_file_list = [self.tr_fp]
        elif dataset.lower() in  ['validation', 'valid', 'va']:
            csv_file_list = [self.va_fp]
        else:
            csv_file_list = [self.ts_fp]
        with graph.as_default():
            with tf.name_scope('dataset'):
                num_cpu = multiprocessing.cpu_count()
                ds = tf.data.TextLineDataset(csv_file_list, ).skip(skip_header_lines)
                ds = ds.shuffle(buffer_size=shuffle_buffer)                
                
                ds = ds.map(lambda row: tf.cast(tf.string_split([row], delimiter=',').values, dtype=tf.string ), num_parallel_calls=num_cpu)
                ds = ds.map(lambda row: [self._decode(row[0],row[1],self.img_dir, flatten), row[2]], num_parallel_calls=num_cpu)
                ds = ds.batch(batch_size)
                ds = ds.prefetch(batch_size*prefetch_batches)
                iter_ds = ds.make_initializable_iterator()
                batch = iter_ds.get_next()
                return batch, iter_ds
