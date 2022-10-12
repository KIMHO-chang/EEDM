from tensorflow.keras.layers import Dense, Input, Dropout, Masking, GRU, Bidirectional, Activation, Lambda, InputLayer, concatenate
from tensorflow.keras import Model


def gru_part(mode, params):
    if mode == 'alphanum':
        input = Input(shape=(params['seq_max_len'], params['alphanum_feature_num']))
        masked_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], params['alphanum_feature_num']))(input)
    if mode == 'special':
        input = Input(shape=(params['seq_max_len'], params['special_feature_num']))
        masked_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], params['special_feature_num']))(input)
    if mode == 'accel':
        input = Input(shape=(params['seq_max_len'], params['accel_feature_num']))
        masked_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], params['accel_feature_num']))(input)
    gru = Bidirectional(GRU(units=params['n_hidden'], return_sequences=False))(masked_input)
    output = Dropout(params['dropout'])(gru)
    model = Model(inputs=input, outputs=output)
    return model


def fusion_part(modes, params):
    input_list = []
    for i in range(len(modes)):
        input = Input(shape=(2 * params['n_hidden'], ))
        input_list.append(input)
    con = concatenate(input_list)
    dense1 = Dense(params['n_classes'] * params['n_latent'], activation='relu', use_bias=True if params['bias'] else False)(con)
    dense2 = Dense(params['n_classes'], use_bias=False)(dense1)
    output = Activation('sigmoid')(dense2) if params['is_clf'] else Activation('linear')(dense2)
    model = Model(inputs=input_list, outputs=output)
    return model


def traditional_deep_mood(modes, params):
    input_list = []
    output_list = []
    for mode in modes:
        input = Input(shape=(params['seq_max_len'], params[mode+'_feature_num']))
        input_list.append(input)
        gru = gru_part(mode, params)(input)
        output_list.append(gru)
    x = concatenate(output_list)
    fusion = fusion_part(modes, params)(x)
    model = Model(inputs=input_list, outputs=fusion)
    return model