import tensorflow as tf
from and_ds import AndData


class Logistic:

    def __init__(self):
        self.graph = tf.Graph()
    
    def compile_model(self, lr, checkpoint_dir, data_dir, batch_size):
        """
        compile summaries,
        add savers

        """
        if not checkpoint_dir[-1] =="/":
            checkpoint_dir+="/"
        with self.graph.as_default():
            self.tr_summaries_dir       = tf.string_join([checkpoint_dir,'train'])
            self.va_summaries_dir       = tf.string_join([checkpoint_dir,'validation'])
            self.tr_checkpoint_prefix   = tf.string_join([checkpoint_dir, 'model.ckpt'])
            self.best_checkpoint_prefix = tf.string_join([checkpoint_dir, 'best_model/model.ckpt'])        

            self.learning_rate = tf.Variable(lr, name='learning rate')
            self.global_step = tf.Variable(0, name='global_step')
            self.load_dataset(data_dir, batch_size)
            self.set_model()
            self.set_optimizer()
            self.loss = tf.reduce_mean( tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.logits), axis=1))
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.set_metric()
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)  
            self.all_summaries      = tf.summary.merge_all()
            self.tr_summary_writer  = tf.summary.FileWriter( self.tr_summaries_dir, self.graph )
            self.val_summary_writer = tf.summary.FileWriter( self.va_summaries_dir, self.graph )
            self.tr_saver           = tf.train.Saver(max_to_keep=10)
            self.best_saver         = tf.train.Saver(max_to_keep=1)
            self.config_proto       = tf.ConfigProto(allow_soft_placement=True)
            self.session            = tf.Session(graph=self.graph, config=self.config_proto )
            self.session.run(tf.global_variables_initializer())
    
    # make it abstract
    def set_model(self):
        with self.graph.as_default():
            self.X = tf.placeholder(name='X', shape=self.input_shape, dtype=tf.float32)
            self.Y = tf.placeholder(name='Y', shape=self.output_shape, dtype=tf.float32)
            
            self.logits = tf.layers.dense(inputs=self.X, units=1, kernel_regularizer= tf.initializers.glorot_uniform() )
            
            self.Y_ = tf.nn.sigmoid(self.logits,)
            self.Y_ = tf.cast(x = tf.greater(tf.cast(self.Y_, tf.float32), tf.constant(0.5, dtype=tf.float32)), dtype=tf.int32, name = 'predictions')
            self.Y = tf.stop_gradient(self.Y)

    # make it abstract
    def load_dataset(self, data_dir_path, batch_size=32,):
        self.andData = AndData(data_dir_path)
        with self.graph.as_default():
            [self.train_X, self.train_Y], self.train_batch_init = self.andData.get_tf_batch(self.graph, batch_size, True, 'train' )
            [self.valid_X, self.valid_Y], self.valid_batch_init = self.andData.get_tf_batch(self.graph, batch_size, True, 'valid' )
            [self.test_X, self.test_Y], self.test_batch_init = self.andData.get_tf_batch(self.graph, batch_size, True, 'test' )
            self.input_shape = self.train_X.shape
            self.output_shape = self.train_Y.shape

    def set_optimizer(self, lr=0.01):
        with self.graph.as_default():
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    def set_metric(self,):
        with self.graph.as_default():
            self.accuracy, self.metric_update = tf.metrics.accuracy(labels=self.Y, predictions=self.Y_, name='f1_score')
            self.metric_init = tf.variables_initializer(tf.get_default_graph().get_collection('local_variables',), 
                                                    name='metrics_initializer')
            

    def train_step(self):
        self.session.run(self.train_batch_init)
        self.session.run(self.metric_init)
        while True:
            try:
                #train model
                tr_X, tr_Y = self.session.run([self.train_X, self.train_Y])
                self.tr_loss, _, self.tr_accuracy, _ , tr_summaries = self.session.run([self.loss, self.metric_update, self.accuracy, self.train_op, self.all_summaries], feed_dict={self.X:tr_X, self.Y:tr_Y})
                print("global_step: {:6}, loss: {:13.6f}, accuracy ={:.6f}".format(self.global_step.eval(self.session), self.tr_loss, self.tr_accuracy), end='\r')
                self.tr_summary_writer.add_summary(tr_summaries, self.global_step.eval(self.session))

            except tf.errors.OutOfRangeError:
                self.tr_saver.save(save_path=self.tr_checkpoint_prefix, sess=self.session, 
                                  global_step=self.global_step.eval(self.session))
                break
    
    def validation_step(self):
        self.session.run(self.valid_batch_init)
        self.session.run(self.metric_init)
        while True:
            try:
                #validate model
                va_X, va_Y = self.session.run([self.valid_X, self.valid_Y])
                self.val_loss, _, self.val_accuracy, val_summaries = self.session.run([self.loss, self.metric_update, 
                                                                           self.accuracy, self.all_summaries],
                                                                         feed_dict={self.X:va_X, self.Y:va_Y})
            except tf.errors.OutOfRangeError:
                self.val_summary_writer.add_summary(val_summaries, self.global_step.eval(self.session))
                print("global_step: {:6}, loss: {:13.6f}, accuracy ={:.6f}, val_loss: {:13.6f}, val_accuracy: {:.6f}".format(self.global_step.eval(self.session), self.tr_loss, self.tr_accuracy, self.val_loss, self.val_accuracy))
                break    
    def test_batch(self):
        pass
    
    def test(self):
        pass

    def train(self, epochs=100, early_stop=True, patience=20):
        with self.graph.as_default():
            for epoc in range(epochs):
                print("epoc: {}".format(epoc) )
                
                # save checkpoint if loss is better than previous best loss
                if float("{:.3f}".format(self.val_loss)) < float("{:.3f}".format(self.best_loss)):
                    self.best_loss = self.val_loss
                    self.cur_patience = 0
                    self.best_saver.save(save_path=self.best_checkpoint_prefix, sess=self.session, 
                                    global_step=self.global_step.eval(self.session))
                else:
                    self.cur_patience += 1

                if cur_patience == patience:
                    print('\n############ Early stopping ############')
                    break


        