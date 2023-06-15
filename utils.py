def sequences2ids(sequence,vectorize_layer):
    return vectorize_layer(sequence)

def ids2sequences(ids,vectorize_layer):
    decode=''
    if type(ids)==int:
        ids=[ids]
    for id in ids:
        decode+=vectorize_layer.get_vocabulary()[id]+' '
    return decode