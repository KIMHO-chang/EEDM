from tensorflow.keras.layers import Dense, Input, Dropout, GRU, Bidirectional, Activation, concatenate
from tensorflow.keras import Model


def gru_part(mode, params):
    if mode == 'alphanum':
        input = Input(shape=(params['seq_max_len'], params['alphanum_feature_num']))
        gru = Bidirectional(GRU(units=params['n_hidden'], return_sequences=False))(input)
        dropout = Dropout(params['dropout'])(gru)
        dense1 = Dense(4, activation='relu')(dropout)
    if mode == 'special':
        input = Input(shape=(params['seq_max_len'], params['special_feature_num']))
        gru = Bidirectional(GRU(units=params['n_hidden'], return_sequences=False))(input)
        dropout = Dropout(params['dropout'])(gru)
        dense1 = Dense(6, activation='relu')(dropout)
    if mode == 'accel':
        input = Input(shape=(params['seq_max_len'], params['accel_feature_num']))
        gru = Bidirectional(GRU(units=params['n_hidden'], return_sequences=False))(input)
        dropout = Dropout(params['dropout'])(gru)
        dense1 = Dense(3, activation='relu')(dropout)
    dense2 = Dense(1)(dense1)
    output = Activation('sigmoid')(dense2) if params['is_clf'] else Activation('linear')(dense2)
    model = Model(inputs=input, outputs=output)
    return model


def fusion_part(params):
    input = Input(shape=(3,))
    dense1 = Dense(params['n_classes'] * params['n_latent'], activation='relu',
                   use_bias=True if params['bias'] else False)(input)
    dense2 = Dense(params['n_classes'], use_bias=False)(dense1)
    output = Activation('sigmoid')(dense2) if params['is_clf'] else Activation('linear')(dense2)
    model = Model(inputs=input, outputs=output)
    return model


def explainable_deep_mood(modes, params):
    input_list = []
    output_list = []
    for mode in modes:
        input = Input(shape=(params['seq_max_len'], params[mode + '_feature_num']))
        input_list.append(input)
        gru = gru_part(mode, params)(input)
        output_list.append(gru)
    x = concatenate(output_list)
    fusion = fusion_part(params)(x)
    model = Model(inputs=input_list, outputs=fusion)
    return model