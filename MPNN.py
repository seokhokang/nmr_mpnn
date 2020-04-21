import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class Model(object):

    def __init__(self, n_node, dim_node, dim_edge, dim_h=50, n_mpnn_step=5, dr=0.1, batch_size=20):

        self.n_node=n_node
        self.dim_node=dim_node
        self.dim_edge=dim_edge

        self.dim_h=dim_h
        self.n_mpnn_step=n_mpnn_step
        self.dr=dr
        self.batch_size=batch_size

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.trn_flag = tf.placeholder(tf.bool)
        
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.dim_node])
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_node, self.n_node, self.dim_edge])  
        self.Y = tf.placeholder(tf.float32, [self.batch_size, self.n_node, 1])
        self.Y_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_node, 1])

        self.hidden_0, self.hidden_n = self._MP(self.batch_size, self.node, self.edge, self.n_mpnn_step, self.dim_h)
        
        self.Y_pred = self._Readout(self.batch_size, self.hidden_0, self.hidden_n, self.dim_h, self.dr)
                 
        # session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()


    def train(self, DV, DE, DY, DM, save_path, frac_val = 0.05):

        def list_to_vec(y):
            vec = np.zeros((self.n_node, 1))
            for i in range(len(y)):       
                if len(y[i])>0: vec[i] = np.mean(y[i])
            
            return vec
        
        DV_trn, DV_val, DE_trn, DE_val, DY_trn, DY_val, DM_trn, DM_val = train_test_split(DV, DE, DY, DM, test_size = frac_val)    
        DY_trn = np.array([list_to_vec(y) for y in DY_trn])

        ## objective function
        reg = tf.square(tf.concat([tf.reshape(v, [-1]) for v in tf.trainable_variables()], 0))
        l2_loss = 1e-10 * tf.reduce_mean(reg)
        
        calib = np.std(DY_trn.flatten()[DM_trn.flatten() > 0])
        
        cost_Y = tf.reduce_sum( tf.abs((self.Y - self.Y_pred) / calib) * self.Y_mask ) / tf.reduce_sum(self.Y_mask)

        vars_MP = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MP')
        vars_Y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y')

        ## configurations
        max_epoch = 500
        n_batch = int(len(DV_trn)/self.batch_size)
        
        lr_list = [1e-3, 1e-4, 1e-5, 1e-6]
        train_op = [tf.train.AdamOptimizer(learning_rate = lr).minimize(cost_Y + l2_loss) for lr in lr_list]
        
        self.sess.run(tf.initializers.global_variables())
        
        ## tranining        
        lr_id = 0
        lr_epoch = 0
        trn_log = np.zeros(max_epoch)
        val_t = np.zeros(max_epoch)
        
        print(':: training')
        
        for epoch in range(max_epoch):

            # training
            [DV_trn, DE_trn, DY_trn, DM_trn] = self._permutation([DV_trn, DE_trn, DY_trn, DM_trn])
            
            trnscores = np.zeros(n_batch)
            for i in range(n_batch):

                start_=i*self.batch_size
                end_=start_+self.batch_size

                trnresult = self.sess.run([train_op[lr_id], cost_Y],
                                          feed_dict = {self.node: DV_trn[start_:end_], self.edge: DE_trn[start_:end_], self.Y: DY_trn[start_:end_],
                                                       self.Y_mask: DM_trn[start_:end_], self.trn_flag: True}) 
                    
                trnscores[i] = trnresult[1]
            
            trn_log[epoch] = np.mean(trnscores) * calib     

            # validation
            val_t[epoch] = self.val_mae(DV_val, DE_val, DY_val, DM_val, 5)  

            print('--training with lr:', lr_list[lr_id], 'epoch id:', epoch, ' trn log:', trn_log[epoch], 'val MAE:', val_t[epoch], 'BEST:', np.min(val_t[0:epoch+1]))

            if np.min(val_t[0:epoch+1]) == val_t[epoch]:
                self.saver.save(self.sess, save_path)

            if epoch - lr_epoch > 10 and np.min(val_t[0:epoch-10]) < np.min(val_t[epoch-10:epoch+1]):
                self.saver.restore(self.sess, save_path)
                lr_epoch = epoch - 0
                lr_id = lr_id + 1
                print('----decrease the learning rate, current BEST: ', self.val_mae(DV_val, DE_val, DY_val, DM_val, 5))
                if lr_id == len(lr_list): break

        print('----termination condition is met')
        self.saver.restore(self.sess, save_path)
        
    
    def val_mae(self, DV, DE, DY, DM, m):
    
        mae = self.test_mae(DV, DE, DY, m)
        
        return mae
    
    
    def test_mae(self, DV, DE, DY, m):
    
        DY_hat = np.mean([self.test(DV, DE) for i in range(m)], 0)

        abs_err = []
        for i, dy in enumerate(DY):
            for j in range(len(dy)):
                if len(dy[j]) > 0: abs_err = abs_err + np.abs(dy[j] - DY_hat[i,j]).tolist()
    
        mae = np.mean( abs_err )
        
        return mae


    def test(self, DV, DE, trn_flag = True):
    
        n_batch = int(len(DV)/self.batch_size)
        DY_hat=[]
        for i in range(n_batch+1):
        
            start_=i*self.batch_size
            end_=start_+self.batch_size

            if len(DV[start_:end_]) == 0:
                continue
                
            elif len(DV[start_:end_]) < self.batch_size:
                c_size = len(DV[start_:end_])
                DY_batch = self.sess.run(self.Y_pred,
                                             feed_dict = {self.node: DV[-self.batch_size:], self.edge: DE[-self.batch_size:], self.trn_flag: trn_flag})
                DY_batch = DY_batch[-c_size:]
                
            else:
                DY_batch = self.sess.run(self.Y_pred,
                                             feed_dict = {self.node: DV[start_:end_], self.edge: DE[start_:end_], self.trn_flag: trn_flag})
                        
            DY_hat.append(DY_batch)
        
        DY_hat = np.concatenate(DY_hat, 0)

        return DY_hat      


    def _permutation(self, set):
    
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
    
        return set
        
    
    def _MP(self, batch_size, node, edge, n_step, hiddendim):

        def _embed_node(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim, activation = tf.nn.tanh)
        
            inp = inp * mask
        
            return inp

        def _edge_nn(inp):
        
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * 5, activation = tf.nn.relu)
            inp = tf.layers.dense(inp, hiddendim * hiddendim)
        
            inp = tf.reshape(inp, [batch_size, self.n_node, self.n_node, hiddendim, hiddendim])
            inp = inp * tf.reshape(1-tf.eye(self.n_node), [1, self.n_node, self.n_node, 1, 1])
            inp = inp * tf.reshape(mask, [batch_size, self.n_node, 1, 1, 1]) * tf.reshape(mask, [batch_size, 1, self.n_node, 1, 1])

            return inp

        def _MPNN(edge_wgt, node_hidden, n_step):
        
            def _msg_nn(wgt, node):
            
                wgt = tf.reshape(wgt, [batch_size * self.n_node, self.n_node * hiddendim, hiddendim])
                node = tf.reshape(node, [batch_size * self.n_node, hiddendim, 1])
            
                msg = tf.matmul(wgt, node)
                msg = tf.reshape(msg, [batch_size, self.n_node, self.n_node, hiddendim])
                msg = tf.transpose(msg, perm = [0, 2, 3, 1])
                msg = tf.reduce_mean(msg, 3)
            
                return msg

            def _update_GRU(msg, node, reuse_GRU):
            
                with tf.variable_scope('mpnn_gru', reuse=reuse_GRU):
            
                    msg = tf.reshape(msg, [batch_size * self.n_node, 1, hiddendim])
                    node = tf.reshape(node, [batch_size * self.n_node, hiddendim])
            
                    cell = tf.nn.rnn_cell.GRUCell(hiddendim)
                    _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)
            
                    node_next = tf.reshape(node_next, [batch_size, self.n_node, hiddendim]) * mask
            
                return node_next

            nhs=[]
            for i in range(n_step):
                message_vec = _msg_nn(edge_wgt, node_hidden)
                node_hidden = _update_GRU(message_vec, node_hidden, reuse_GRU=(i!=0))
                nhs.append(node_hidden)
        
            out = tf.concat(nhs, axis=2)
            
            return out

        with tf.variable_scope('MP', reuse=False):
        
            mask = tf.clip_by_value(tf.reduce_max(node, 2, keepdims=True), 0, 1)
            
            edge_wgt = _edge_nn(edge)
            hidden_0 = _embed_node(node)
            hidden_n = _MPNN(edge_wgt, hidden_0, n_step)
            
        return hidden_0, hidden_n


    def _Readout(self, batch_size, hidden_0, hidden_n, hiddendim, drate):

        with tf.variable_scope('Y', reuse=False):
            
            inp = tf.concat([hidden_0, hidden_n], 2)
            
            inp = tf.layers.dense(inp, hiddendim * 10, activation = tf.nn.relu)
            inp = tf.layers.dropout(inp, drate, training = self.trn_flag)
            inp = tf.layers.dense(inp, hiddendim * 10, activation = tf.nn.relu)
            inp = tf.layers.dropout(inp, drate, training = self.trn_flag)
            inp = tf.layers.dense(inp, hiddendim * 10, activation = tf.nn.relu)
            
            rout = tf.layers.dense(inp, 1)

        return rout
