"""
@author sourabhxiii
"""

import numpy as np
import keras

BATCH_SIZE = 128
EPOCHS = 50
VALIDATION_SPLIT = 0.1


class EmoconModel():
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
            
        
        with keras.backend.name_scope('indep_seq_proc'):
            ts1 = keras.layers.Bidirectional(
                keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(embedded_turn1)
            # ts1 = keras.layers.Bidirectional(
            #     keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(ts1)

            ts2 = keras.layers.Bidirectional(
                keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(embedded_turn2)
            # ts2 = keras.layers.Bidirectional(
            #     keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(ts2)
            
            ts3 = keras.layers.Bidirectional(
                keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(embedded_turn3)
            # ts3 = keras.layers.Bidirectional(
            #     keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))(ts3)
        

        with keras.backend.name_scope('indep_conv_proc'):
            tc1 = keras.layers.Conv1D(filters=32,
                    kernel_size=5,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedded_turn1)
            tc1 = keras.layers.GlobalMaxPooling1D()(tc1)
            tc1 = keras.layers.Dense(32)(tc1)
            tc1 = keras.layers.PReLU()(tc1)
            tc1 = keras.layers.Dropout(0.2)(tc1)
            tc1 = keras.layers.BatchNormalization()(tc1)

            tc2 = keras.layers.Conv1D(filters=32,
                    kernel_size=5,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedded_turn2)
            tc2 = keras.layers.GlobalMaxPooling1D()(tc2)
            tc2 = keras.layers.Dense(32)(tc2)
            tc2 = keras.layers.PReLU()(tc2)
            tc2 = keras.layers.Dropout(0.2)(tc2)
            tc2 = keras.layers.BatchNormalization()(tc2)

            tc3 = keras.layers.Conv1D(filters=32,
                    kernel_size=5,
                    padding='valid',
                    activation='relu',
                    strides=1)(embedded_turn3)
            tc3 = keras.layers.GlobalMaxPooling1D()(tc3)
            tc3 = keras.layers.Dense(32)(tc3)
            tc3 = keras.layers.PReLU()(tc3)
            tc3 = keras.layers.Dropout(0.2)(tc3)
            tc3 = keras.layers.BatchNormalization()(tc3)

        with keras.backend.name_scope('concat'):
            x = keras.layers.Concatenate()([ts1, ts2, ts3, tc1, tc2, tc3])

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

        model = keras.models.Model([input_turn1, input_turn2, input_turn3], preds)

        # set up callbacks
        filepath = 'model.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
        chkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1
            , save_best_only=True, save_weights_only=False, mode='auto', period=1)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5
            , verbose=1, mode='auto', baseline=None)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=3, min_lr=0.0001, verbose=1)

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