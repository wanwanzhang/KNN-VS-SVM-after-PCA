import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import time

if __name__ =="__main__":
    train_num = 20000
    test_num = 30000
    data = pd.read_csv('C:\\Users\\Wanwan Zhang\\Desktop\\2016FALL\\ADS\\train.csv')
    train_data = data.values[0:train_num,1:]
    train_label = data.values[0:train_num,0]
    test_data = data.values[train_num:test_num,1:]
    test_label = data.values[train_num:test_num,0]

    t = time.time()
    pca=PCA(n_components = 0.8)
    train_x = pca.fit_transform(train_data)
    test_x = pca.transform(test_data)
    neighbors = KNeighborsClassifier(n_neighbors=4)
    neighbors.fit(train_x,train_label)
    pre= neighbors.predict(test_x)

    acc = float((pre==test_label).sum())/len(test_x)
    print ('first accuracy rate：%f,time spent：%.2fs' %(acc,time.time()-t))

    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn import svm
    import time

    if __name__ == "__main__":
        train_num = 5000
        test_num = 7000
        data = pd.read_csv('C:\\Users\\Wanwan Zhang\\Desktop\\2016FALL\\ADS\\train.csv')
        train_data = data.values[0:train_num, 1:]
        train_label = data.values[0:train_num, 0]
        test_data = data.values[train_num:test_num, 1:]
        test_label = data.values[train_num:test_num, 0]
        t = time.time()
        # svm
        pca = PCA(n_components=0.8, whiten=True)
        train_x = pca.fit_transform(train_data)
        test_x = pca.transform(test_data)
        svc = svm.SVC(kernel='rbf', C=10)
        svc.fit(train_x, train_label)
        pre = svc.predict(test_x)
        acc = float((pre == test_label).sum()) / len(test_x)
        print
        'second accuracy rate：%f,time spent：%.2fs' % (acc, time.time() - t)