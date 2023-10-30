from sklearn.datasets import fetch_20newsgroups

newsgroup_train = fetch_20newsgroups(subset = 'train', remove = ('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset= "test", remove = ('headers', 'footers', 'quotes'))
def fetch_training_data():
    print("3. Fetch data fc")
    # newsgroup_train = fetch_20newsgroup_trains(subset = 'train', remove = ('headers', 'footers', 'quotes'))
    training_data = newsgroup_train.data
    target = newsgroup_train.target
    return{"training_data": training_data, "target": target}
    # return (training_data, target)

def fetch_testing_data():
    testing_data = newsgroups_test.data
    target = newsgroup_train.target
    return{"testing_data": testing_data, "target": target}
    
def fetch_target_names():
    target_names = newsgroup_train.traget_names
    return target_names

def fetch_training_labels():
    labels = newsgroup_train.target
    return labels

def fetch_test_labels():
    labels = newsgroups_test.target
    return labels
