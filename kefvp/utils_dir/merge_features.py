import pickle
import numpy as np

def merge_features(base_dir, duration, source, mode):
    '''
        merge sequence: train, test, dev
    '''
    
    trainft = pickle.load(open(base_dir + '/feaures_train_{}days_{}_{}.pkl'.format(str(duration), source, mode), 'rb'))
    testft = pickle.load(open(base_dir + '/feaures_test_{}days_{}_{}.pkl'.format(str(duration), source, mode), 'rb'))
    devft = pickle.load(open(base_dir + '/feaures_dev_{}days_{}_{}.pkl'.format(str(duration), source, mode), 'rb'))
    
    finalft = np.array(trainft + testft + devft).squeeze(1)
    
    finalft.dump(open(base_dir + '/final{}days_{}_{}.npy'.format(str(duration), source, mode), 'wb'))


if __name__ == '__main__':
    
    # base_dir = '/home/niuhao/project/DocTime/Earning_call/html_www2020/save_features'
    # base_dir = '/home/niuhao/project/DocTime/Earning_call/html_www2020/save_features/price_reviseautofm'
    base_dir = '/home/niuhao/project/DocTime/Earning_call/html_www2020/save_features/text_audio_fuse_feature'
    
    # merge_features(base_dir, duration=3, source='text', mode='reg')  
    # merge_features(base_dir, duration=3, source='text&audio', mode='reg')
    # merge_features(base_dir, duration=3, source='audio', mode='reg')
    # merge_features(base_dir, duration=3, source='price', mode='reg')
    # merge_features(base_dir, duration=3, source='text_audio', mode='reg')
    # merge_features(base_dir, duration=3, source='price_seasonal', mode='reg')
    # merge_features(base_dir, duration=3, source='price_reviseautofm', mode='reg')
    merge_features(base_dir, duration=3, source='text_audio_fuse_feature', mode='reg')