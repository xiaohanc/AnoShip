import pickle as pk

def getEntityList(path='./dataset/', dataset='SWaT', entities='all'):

    if entities != 'all':
        return [int(x) for x in entities.split(',')]
    else:
        with open(path + dataset + '/' + dataset + '.pk', 'rb') as file:
            data = pk.load(file)
        entity_num = len(data['init_data'])
        return list(range(entity_num))

