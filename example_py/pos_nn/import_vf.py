import pickle

with open("safety_VF.pkl", 'rb') as f:
    data = pickle.load(f)

with open('test_2.pkl', 'wb') as f:
     pickle.dump(data, f, protocol=0)