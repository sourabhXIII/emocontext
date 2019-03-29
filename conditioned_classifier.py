"""
@author sourabhxiii
"""

import numpy as np
import keras
from f1callback import F1Value
from attention_layer import Attention
BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.1


class ConditionedModel():
    def __init__(self, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        , output_dim):
        with keras.backend.name_scope('embedding'):
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        cond_GRU = keras.layers.GRU(128, return_sequences=True, dropout=0.2)

        with keras.backend.name_scope('seq_input'):
            input_turn1 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn1')
            embedded_turn1 = embedding_layer(input_turn1)
            
            input_turn2 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn2')
            embedded_turn2 = embedding_layer(input_turn2)

            input_turn3 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn3')
            embedded_turn3 = embedding_layer(input_turn3)

            # keras.layers.GRU(128
            #     , kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))

        with keras.backend.name_scope('turn1_proc'):
            ts1 = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))(embedded_turn1)
            ts1 = keras.layers.Dropout(0.3)(ts1)
            ts1, ts1_h = \
                        keras.layers.GRU(128, return_state=True
                            , dropout=0.2)(ts1)
        
        with keras.backend.name_scope('turn2_proc'):
            ts2 = cond_GRU(embedded_turn2, initial_state=[ts1_h])
            ts2 = keras.layers.Dropout(0.3)(ts2)
            ts2, ts2_h = \
                keras.layers.GRU(128, return_state=True
                    , dropout=0.2)(ts2)

        with keras.backend.name_scope('turn3_proc'):
            ts3 = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))(embedded_turn3)
            ts3 = keras.layers.Dropout(0.3)(ts3)
            ts3 = keras.layers.Bidirectional(
                        keras.layers.GRU(128, return_state=False
                            , dropout=0.2))(ts3)

        with keras.backend.name_scope('concat'):
            merged = keras.layers.Concatenate()([ts2, ts3])

        with keras.backend.name_scope('common'):
            x = keras.layers.Dense(512)(merged)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(256)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(64)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)

        preds = keras.layers.Dense(output_dim, activation='softmax')(x)

        model = keras.models.Model([input_turn1, input_turn2, input_turn3], preds)

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=2, min_lr=0.0001, verbose=1)

        self.callback_list=[keras.callbacks.History(), chkpoint, reduce_lr]


        opt = keras.optimizers.Adam(0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc']
        )

        self.model = model

        # Summarize the model
        print(self.model.summary())

    def train(self, t1_train, t1_test, t2_train
        , t2_test, t3_train, t3_test, target_train
        , target_test, class_weights):
        print('Training model...')
        hist = self.model.fit(
            [t1_train, t2_train, t3_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([t1_test, t2_test, t3_test], target_test),
            verbose=1,
            class_weight=class_weights,
            callbacks=self.callback_list,
            shuffle=True
            )
        return hist

    def eval(self, t1_test, t2_test, t3_test, target_test):
        print('Evaluating model...')
        hist = self.model.evaluate(
            [t1_test, t2_test, t3_test],
            target_test
        )



class TextualFeatureModel():
    def __init__(self, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        , output_dim):
        with keras.backend.name_scope('embedding'):
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        with keras.backend.name_scope('seq_input'):
            input_turn1 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn1')
            embedded_turn1 = embedding_layer(input_turn1)
            
            input_turn2 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn2')
            embedded_turn2 = embedding_layer(input_turn2)

            input_turn3 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn3')
            embedded_turn3 = embedding_layer(input_turn3)
        
        with keras.backend.name_scope('conv_t1'):
            tc1 = keras.layers.Conv1D(64, 3, data_format='channels_first')(embedded_turn1)
            tc1 = keras.layers.Conv1D(64, 4, data_format='channels_first')(tc1)
            tc1 = keras.layers.Conv1D(64, 5, data_format='channels_first')(tc1)
            tc1 = keras.layers.MaxPooling1D(3)(tc1)
            tc1 = keras.layers.PReLU()(tc1)
        with keras.backend.name_scope('conv_t2'):
            tc2 = keras.layers.Conv1D(64, 3, data_format='channels_first')(embedded_turn2)
            tc2 = keras.layers.Conv1D(64, 4, data_format='channels_first')(tc2)
            tc2 = keras.layers.Conv1D(64, 5, data_format='channels_first')(tc2)
            tc2 = keras.layers.MaxPooling1D(3)(tc2)
            tc2 = keras.layers.PReLU()(tc2)
        with keras.backend.name_scope('conv_t3'):
            tc3 = keras.layers.Conv1D(64, 3, data_format='channels_first')(embedded_turn3)
            tc3 = keras.layers.Conv1D(64, 4, data_format='channels_first')(tc3)
            tc3 = keras.layers.Conv1D(64, 5, data_format='channels_first')(tc3)
            tc3 = keras.layers.MaxPooling1D(3)(tc3)
            tc3 = keras.layers.PReLU()(tc3)
        
        with keras.backend.name_scope('concat'):
            c = keras.layers.Concatenate()([tc1, tc2, tc3])
        
        with keras.backend.name_scope('common'):
            x = keras.layers.Dense(512)(c)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(256)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(64)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Flatten()(x)


            preds = keras.layers.Dense(output_dim, activation='softmax')(x)

        model = keras.models.Model([input_turn1, input_turn2, input_turn3], preds)

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=2, min_lr=0.0001, verbose=1)

        self.callback_list=[keras.callbacks.History(), chkpoint, es, reduce_lr]


        opt = keras.optimizers.Adam(0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc']
        )

        self.model = model

        # Summarize the model
        print(self.model.summary())

    def train(self, turn1, turn2, turn3, targets, class_weights):
        print('Training model...')
        hist = self.model.fit(
            [turn1, turn2, turn3],
            targets,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            verbose=1,
            class_weight=class_weights,
            callbacks=self.callback_list,
            shuffle=True
            )
        return hist



class SimpleClassifier():
    def __init__(self, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        , output_dim):
        with keras.backend.name_scope('embedding'):
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        with keras.backend.name_scope('seq_input'):
            input_conv = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn1')
            embedded_conv = embedding_layer(input_conv)
        
        with keras.backend.name_scope('seq_input'):
            x = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True))(embedded_conv)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=False))(x)
            x = keras.layers.Dropout(0.2)(x)

        with keras.backend.name_scope('common'):
            x = keras.layers.Dense(512)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(256)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(64)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)

        preds = keras.layers.Dense(output_dim, activation='softmax')(x)

        model = keras.models.Model(input_conv, preds)

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=2, min_lr=0.0001, verbose=1)

        self.callback_list=[keras.callbacks.History(), chkpoint, es, reduce_lr]


        opt = keras.optimizers.Adam(0.001)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc']
        )

        self.model = model

        # Summarize the model
        print(self.model.summary())

    def train(self, conv_train, conv_test, target_train
        , target_test, class_weights):
        print('Training model...')
        hist = self.model.fit(
            conv_train,
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(conv_test, target_test),
            verbose=1,
            class_weight=class_weights,
            callbacks=self.callback_list,
            shuffle=True
            )
        return hist


class ConditionedModelV1():
    def __init__(self, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        , output_dim):
        with keras.backend.name_scope('embedding'):
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        cond_GRU = keras.layers.GRU(128, return_sequences=True)

        with keras.backend.name_scope('seq_input'):
            input_turn1 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn1')
            embedded_turn1 = embedding_layer(input_turn1)
            
            input_turn2 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn2')
            embedded_turn2 = embedding_layer(input_turn2)

            input_turn3 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn3')
            embedded_turn3 = embedding_layer(input_turn3)

            # keras.layers.GRU(128
            #     , kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))

        with keras.backend.name_scope('turn1_proc'):
            ts1 = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))(embedded_turn1)
            ts1 = keras.layers.Dropout(0.3)(ts1)
            ts1, ts1_h = \
                        keras.layers.GRU(128, return_state=True
                            , dropout=0.2)(ts1)
        
        with keras.backend.name_scope('turn2_proc'):
            ts2 = cond_GRU(embedded_turn2, initial_state=[ts1_h])
            ts2 = keras.layers.Dropout(0.3)(ts2)
            ts2, ts2_h = \
                keras.layers.GRU(128, return_state=True
                    , dropout=0.2)(ts2)

        with keras.backend.name_scope('turn3_proc'):
            ts3 = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))(embedded_turn3)
            ts3 = keras.layers.Dropout(0.3)(ts3)
            ts3 = keras.layers.Bidirectional(
                        keras.layers.GRU(256, return_sequences=True
                            , return_state=False
                            , dropout=0.2))(ts3)
        
        # with keras.backend.name_scope('concat1'):
        #     merged = keras.layers.Concatenate()([ts1, ts2, ts3])

        with keras.backend.name_scope('attn'):
            a = Attention(MAX_SEQUENCE_LENGTH)(ts3)
        with keras.backend.name_scope('common'):
            x = keras.layers.Dense(128)(a)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x1 = keras.layers.Dense(128)(x)
            x1 = keras.layers.PReLU()(x1)
            x1 = keras.layers.Dropout(0.5)(x1)
            x1 = keras.layers.BatchNormalization()(x1)
            x1 = keras.layers.Add()([x, x1])

            x2 = keras.layers.Dense(128)(x1)
            x2 = keras.layers.PReLU()(x2)
            x2 = keras.layers.Dropout(0.25)(x2)
            x2 = keras.layers.BatchNormalization()(x2)
            x2 = keras.layers.Add()([x1, x2])

            x3 = keras.layers.Dense(128)(x2)
            x3 = keras.layers.PReLU()(x3)
            x3 = keras.layers.Dropout(0.5)(x3)
            x3 = keras.layers.BatchNormalization()(x3)
            x3 = keras.layers.Add()([x2, x3])
            
            x4 = keras.layers.Dense(output_dim)(x3)
            x4 = keras.layers.PReLU()(x4)
            x4 = keras.layers.BatchNormalization()(x4)


        with keras.backend.name_scope('woe_input'):
            t1_woe = keras.layers.Input(shape=(4,), name='t1_woe')
            t2_woe = keras.layers.Input(shape=(4,), name='t2_woe')
            t3_woe = keras.layers.Input(shape=(4,), name='t3_woe')

        with keras.backend.name_scope('concat2'):
            merged = keras.layers.Concatenate()([x4, t1_woe, t2_woe, t3_woe])

        with keras.backend.name_scope('process_woe'):
            x = keras.layers.Dense(32)(merged)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x1 = keras.layers.Dense(32)(x)
            x1 = keras.layers.PReLU()(x1)
            x1 = keras.layers.Dropout(0.5)(x1)
            x1 = keras.layers.BatchNormalization()(x1)
            x1 = keras.layers.Add()([x, x1])

            x2 = keras.layers.Dense(32)(x1)
            x2 = keras.layers.PReLU()(x2)
            x2 = keras.layers.Dropout(0.5)(x2)
            x2 = keras.layers.BatchNormalization()(x2)
            x2 = keras.layers.Add()([x1, x2])

        preds = keras.layers.Dense(output_dim, activation='softmax')(x2)

        model = keras.models.Model(
            [input_turn1, input_turn2, input_turn3, t1_woe, t2_woe, t3_woe], preds)

        opt = keras.optimizers.Adam(0.001)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc']
        )
        self.model = model

        # Summarize the model
        print(self.model.summary())
        from keras.utils import plot_model
        plot_model(model, to_file='condv1.png', show_shapes=True)

    def train(self, t1_train, t1_test, t2_train, t2_test
        , t3_train, t3_test, t1_woe_train, t1_woe_test
        , t2_woe_train, t2_woe_test, t3_woe_train, t3_woe_test
        , target_train, target_test, class_weights):
        print('Training model...')

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)

        import warmup_cosine_lr_decay_scheduler as wcos_lr_sch
        import math
        # number of warmup epochs
        warmup_epoch = 2
        # base learning rate after warmup.
        learning_rate_base = 0.01
        # total number of steps (NumEpoch * StepsPerEpoch)
        print("input shape {}".format(t1_train.shape))
        STEP_SIZE_TRAIN = math.ceil(t1_train.shape[0]//BATCH_SIZE)
        total_steps = int(EPOCHS * STEP_SIZE_TRAIN)
        # compute the number of warmup batches.
        warmup_steps = int(warmup_epoch * STEP_SIZE_TRAIN)
        # how many steps to hold the base lr
        hold_base_rate_epoch = 50
        hold_base_rate_steps = int(hold_base_rate_epoch * STEP_SIZE_TRAIN)
        # create the Learning rate scheduler.
        warm_up_lr = wcos_lr_sch.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                total_steps=total_steps,
                                                warmup_learning_rate=0.0001,
                                                warmup_steps=warmup_steps,
                                                hold_base_rate_steps=hold_base_rate_steps,
                                                verbose=1)
        self.callback_list=[keras.callbacks.History(), chkpoint, warm_up_lr]

        hist = self.model.fit(
            [t1_train, t2_train, t3_train, t1_woe_train, t2_woe_train, t3_woe_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([t1_test, t2_test, t3_test, t1_woe_test, t2_woe_test, t3_woe_test], target_test),
            verbose=1,
            class_weight=class_weights,
            callbacks=self.callback_list,
            shuffle=True
            )
        return hist

    def eval(self, t1_test, t2_test, t3_test, target_test):
        print('Evaluating model...')
        hist = self.model.evaluate(
            [t1_test, t2_test, t3_test],
            target_test
        )



class ConditionedModelV2():
    def __init__(self, embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH
        , output_dim):
        with keras.backend.name_scope('embedding'):
            # load pre-trained word embeddings into an Embedding layer
            embedding_layer = keras.layers.Embedding(input_dim=embedding_matrix.shape[0]
                , output_dim=EMBEDDING_DIM
                , embeddings_initializer=keras.initializers.Constant(embedding_matrix)
                , input_length=MAX_SEQUENCE_LENGTH
                , trainable=False
                , name='Embedding'
                )
        cond_GRU = keras.layers.GRU(128, return_sequences=True)

        with keras.backend.name_scope('seq_input'):
            input_turn1 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn1')
            embedded_turn1 = embedding_layer(input_turn1)
            
            input_turn2 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn2')
            embedded_turn2 = embedding_layer(input_turn2)

            input_turn3 = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='turn3')
            embedded_turn3 = embedding_layer(input_turn3)

            # keras.layers.GRU(128
            #     , kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))

        with keras.backend.name_scope('turn1_proc'):
            ts1 = keras.layers.Bidirectional(
                    keras.layers.GRU(128, return_sequences=True
                        , dropout=0.2))(embedded_turn1)
            ts1 = keras.layers.Dropout(0.3)(ts1)
            ts1, ts1_h = \
                        keras.layers.GRU(128, return_state=True
                            , dropout=0.2)(ts1)
        self.t1_model = keras.models.Model(inputs=input_turn1, outputs=ts1_h, name='t1_model')
        print("Turn 1 Model:")
        print(self.t1_model.summary())
        
        with keras.backend.name_scope('turn2_proc'):
            latent_input = keras.layers.Input(shape=(128,), name='t2_latent_input')
            ts2 = cond_GRU(embedded_turn2, initial_state=[latent_input])
            ts2 = keras.layers.Dropout(0.3)(ts2)
            ts2, ts2_h = \
                keras.layers.GRU(128, return_state=True
                    , dropout=0.2)(ts2)
        self.t2_model = keras.models.Model(inputs=[input_turn2, latent_input], outputs=ts2_h, name='t2_model')
        print("Turn 2 Model:")
        print(self.t2_model.summary())

        with keras.backend.name_scope('turn3_proc'):
            latent_input = keras.layers.Input(shape=(128,), name='t3_latent_input')
            ts3 = cond_GRU(embedded_turn3, initial_state=[latent_input])
            ts3 = keras.layers.Dropout(0.3)(ts3)
            ts3 = keras.layers.GRU(256, return_sequences=False
                            , return_state=False
                            , dropout=0.2)(ts3)
        self.t3_model = keras.models.Model(inputs=[input_turn3, latent_input], outputs=ts3, name='t3_model')
        print("Turn 3 Model:")
        print(self.t3_model.summary())

        latent_input = keras.layers.Input(shape=(256,), name='conv_latent_input')
        with keras.backend.name_scope('common'):
            x = keras.layers.Dense(128)(latent_input)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(128)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.25)(x)
            x = keras.layers.BatchNormalization()(x)
            
            x = keras.layers.Dense(output_dim)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.BatchNormalization()(x)


        with keras.backend.name_scope('woe_input'):
            t1_woe = keras.layers.Input(shape=(4,), name='t1_woe')
            t2_woe = keras.layers.Input(shape=(4,), name='t2_woe')
            t3_woe = keras.layers.Input(shape=(4,), name='t3_woe')

        with keras.backend.name_scope('concat2'):
            merged = keras.layers.Concatenate()([x, t1_woe, t2_woe, t3_woe])

        with keras.backend.name_scope('process_woe'):
            x = keras.layers.Dense(32)(merged)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Dense(32)(x)
            x = keras.layers.PReLU()(x)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.BatchNormalization()(x)

        preds = keras.layers.Dense(output_dim, activation='softmax')(x)

        model = keras.models.Model(
            [latent_input, t1_woe, t2_woe, t3_woe], preds, name='complete_model')

        opt = keras.optimizers.Adam(0.001)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc']
        )
        self.model = model

        # Summarize the model
        print("Complete Model:")
        print(self.model.summary())
        from keras.utils import plot_model
        plot_model(model, to_file='condv2.png', show_shapes=True)

    def train(self, t1_train, t1_test, t2_train, t2_test
        , t3_train, t3_test, t1_woe_train, t1_woe_test
        , t2_woe_train, t2_woe_test, t3_woe_train, t3_woe_test
        , target_train, target_test, class_weights):
        print('Training model...')

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)

        import warmup_cosine_lr_decay_scheduler as wcos_lr_sch
        import math
        # number of warmup epochs
        warmup_epoch = 2
        # base learning rate after warmup.
        learning_rate_base = 0.01
        # total number of steps (NumEpoch * StepsPerEpoch)
        print("input shape {}".format(t1_train.shape))
        STEP_SIZE_TRAIN = math.ceil(t1_train.shape[0]//BATCH_SIZE)
        total_steps = int(EPOCHS * STEP_SIZE_TRAIN)
        # compute the number of warmup batches.
        warmup_steps = int(warmup_epoch * STEP_SIZE_TRAIN)
        # how many steps to hold the base lr
        hold_base_rate_epoch = 50
        hold_base_rate_steps = int(hold_base_rate_epoch * STEP_SIZE_TRAIN)
        # create the Learning rate scheduler.
        warm_up_lr = wcos_lr_sch.WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                total_steps=total_steps,
                                                warmup_learning_rate=0.0001,
                                                warmup_steps=warmup_steps,
                                                hold_base_rate_steps=hold_base_rate_steps,
                                                verbose=1)
        self.callback_list=[keras.callbacks.History(), chkpoint, warm_up_lr]


        self.t2_model()
        hist = self.model.fit(
            [t1_train, t2_train, t3_train, t1_woe_train, t2_woe_train, t3_woe_train],
            target_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([t1_test, t2_test, t3_test, t1_woe_test, t2_woe_test, t3_woe_test], target_test),
            verbose=1,
            class_weight=class_weights,
            callbacks=self.callback_list,
            shuffle=True
            )
        return hist

    def eval(self, t1_test, t2_test, t3_test, target_test):
        print('Evaluating model...')
        hist = self.model.evaluate(
            [t1_test, t2_test, t3_test],
            target_test
        )


